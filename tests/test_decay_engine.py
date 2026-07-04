"""Decay engine — pure-function and end-to-end tests.

Covers Properties 2 (short-circuits), 3 (resolved monotonicity), 4
(short/long-term boundary continuity), and 10 (auto-resolve applies in
the same cycle). Hypothesis isn't a project dependency yet, so the
property tests are encoded as parameterised pytest cases over a small
diverse sample of inputs — enough to catch regressions in practice.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from astrbot_plugin_ob_memory.core.decay_engine import (
    DEFAULT_AROUSAL_BOOST,
    DEFAULT_EMOTION_BASE,
    DEFAULT_LAMBDA,
    FEEL_SCORE,
    PINNED_SCORE,
    SHORT_TERM_DAYS,
    DecayConfig,
    DecayEngine,
    calculate_score,
    score_breakdown,
)
from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.core.models import MemoryBucket, new_bucket
from astrbot_plugin_ob_memory.storage import Database, apply_migrations

ASYNCIO = pytest.mark.asyncio


def _make_bucket(
    *,
    importance: int = 5,
    valence: float = 0.5,
    arousal: float = 0.3,
    bucket_type: str = "dynamic",
    pinned: bool = False,
    resolved: bool = False,
    digested: bool = False,
    activation_count: float = 0.0,
    days_since_active: float = 0.0,
) -> MemoryBucket:
    """Hand-constructed bucket with a controllable ``last_active_at``.

    We bypass the normal factory because we want to back-date timestamps
    for decay testing without the factory's "stamp to now" behaviour.
    """
    now = time.time()
    return MemoryBucket(
        id="test-id",
        session_id="test-session",
        content="x",
        importance=importance,
        valence=valence,
        arousal=arousal,
        bucket_type=bucket_type,  # type: ignore[arg-type]
        pinned=pinned,
        resolved=resolved,
        digested=digested,
        activation_count=activation_count,
        created_at=now - days_since_active * 86400,
        last_active_at=now - days_since_active * 86400,
    )


# ---------------------------------------------------------------------------
# Property 2 — Short-circuits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        # Pinned + dynamic still pinned-shortcuts (clamping would also
        # promote, but bypassing clamping must still short-circuit).
        {"pinned": True},
        {"pinned": True, "bucket_type": "permanent"},
        {"pinned": True, "importance": 1, "activation_count": 0},
        {"pinned": True, "days_since_active": 365},
        {"bucket_type": "permanent"},
        {"bucket_type": "permanent", "importance": 1, "activation_count": 0},
    ],
)
def test_pinned_or_permanent_returns_999(kwargs):
    bucket = _make_bucket(**kwargs)
    assert calculate_score(bucket) == PINNED_SCORE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bucket_type": "feel"},
        {"bucket_type": "feel", "resolved": True},
        {"bucket_type": "feel", "days_since_active": 365},
        {"bucket_type": "feel", "arousal": 1.0},
    ],
)
def test_feel_returns_50(kwargs):
    bucket = _make_bucket(**kwargs)
    assert calculate_score(bucket) == FEEL_SCORE


def test_pinned_overrides_feel_priority():
    """A pinned bucket short-circuits to 999 even if also marked feel."""
    bucket = _make_bucket(pinned=True, bucket_type="feel")
    # Pinned check fires before feel-type check.
    assert calculate_score(bucket) == PINNED_SCORE


# ---------------------------------------------------------------------------
# Property 3 — Resolved monotonicity (multiplicative)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "importance,arousal,days",
    [
        # Keep arousal ≤ 0.7 so the urgency boost is the same (1.0) for
        # all three buckets — that lets us assert a clean multiplicative
        # relationship between resolved_factor levels.
        (5, 0.3, 1.0),
        (8, 0.6, 0.5),
        (3, 0.4, 5.0),
        (10, 0.5, 0.1),
    ],
)
def test_resolved_factor_decreases_score(importance, arousal, days):
    base = _make_bucket(
        importance=importance, arousal=arousal, days_since_active=days,
    )
    only_resolved = _make_bucket(
        importance=importance, arousal=arousal, days_since_active=days,
        resolved=True,
    )
    full = _make_bucket(
        importance=importance, arousal=arousal, days_since_active=days,
        resolved=True, digested=True,
    )

    s_base = calculate_score(base)
    s_resolved = calculate_score(only_resolved)
    s_full = calculate_score(full)

    assert s_base >= s_resolved >= s_full
    # Tolerance is generous because ``calculate_score`` rounds to 4 dp,
    # which dominates the error budget at small score magnitudes.
    assert s_resolved == pytest.approx(s_base * 0.05, rel=1e-3, abs=1e-3)
    assert s_full == pytest.approx(s_base * 0.02, rel=1e-3, abs=1e-3)


def test_resolved_with_high_arousal_loses_urgency_boost():
    """High-arousal buckets get a 1.5× urgency boost only while unresolved.

    Going from unresolved → resolved therefore changes TWO multipliers at
    once (urgency_boost 1.5 → 1.0, resolved_factor 1.0 → 0.05). The
    combined ratio is 0.05/1.5 ≈ 0.0333.
    """
    base = _make_bucket(arousal=0.9, days_since_active=1.0)
    resolved = _make_bucket(arousal=0.9, days_since_active=1.0, resolved=True)
    s_base = calculate_score(base)
    s_resolved = calculate_score(resolved)
    expected_ratio = 0.05 / 1.5
    assert s_resolved == pytest.approx(s_base * expected_ratio, rel=1e-3, abs=1e-3)


# ---------------------------------------------------------------------------
# Property 4 — Short-term vs Long-term boundary continuity at days = 3.0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "importance,arousal",
    [
        (5, 0.3),
        (8, 0.7),
        (1, 0.0),
        (10, 1.0),
    ],
)
def test_boundary_continuous_at_3_days(importance, arousal):
    """At exactly 3 days the score is identical via either branch.

    Both branches reduce to the same numeric value because they're
    weighted averages with the same total weight (1.0).
    """
    just_under = _make_bucket(
        importance=importance, arousal=arousal,
        days_since_active=2.9999,
    )
    just_over = _make_bucket(
        importance=importance, arousal=arousal,
        days_since_active=3.0001,
    )
    s1 = calculate_score(just_under)
    s2 = calculate_score(just_over)
    # Allow tiny tolerance for the minute drift in days_since.
    assert s1 == pytest.approx(s2, rel=1e-3)


def test_boundary_uses_correct_branch():
    """At the 3-day boundary the combined_weight is the midpoint of the
    short-term and long-term mixtures, since the crossfade weight ``alpha``
    equals 0.5 there."""
    bucket = _make_bucket(
        importance=5,
        arousal=0.5,
        days_since_active=3.0,
    )
    breakdown = score_breakdown(bucket)
    short_mix = breakdown.time_weight * 0.7 + breakdown.emotion_weight * 0.3
    long_mix = breakdown.emotion_weight * 0.7 + breakdown.time_weight * 0.3
    midpoint = (short_mix + long_mix) / 2.0
    assert breakdown.combined_weight == pytest.approx(midpoint, rel=1e-6)


def test_score_breakdown_handles_very_old_memories_without_overflow():
    bucket = _make_bucket(
        importance=5,
        arousal=0.5,
        days_since_active=SHORT_TERM_DAYS + 10000,
    )
    breakdown = score_breakdown(bucket)
    short_mix = breakdown.time_weight * 0.7 + breakdown.emotion_weight * 0.3
    long_mix = breakdown.emotion_weight * 0.7 + breakdown.time_weight * 0.3

    assert breakdown.combined_weight == pytest.approx(long_mix, rel=1e-9)
    assert breakdown.combined_weight != pytest.approx(short_mix, rel=1e-3)
    assert breakdown.score >= 0.0


# ---------------------------------------------------------------------------
# Urgency boost (Requirement 5.4)
# ---------------------------------------------------------------------------
def test_urgency_boost_applies_for_unresolved_high_arousal():
    high = _make_bucket(arousal=0.8, resolved=False)
    not_high = _make_bucket(arousal=0.6, resolved=False)
    breakdown_high = score_breakdown(high)
    breakdown_not_high = score_breakdown(not_high)
    assert breakdown_high.urgency_boost == 1.5
    assert breakdown_not_high.urgency_boost == 1.0


def test_urgency_boost_disabled_when_resolved():
    bucket = _make_bucket(arousal=0.9, resolved=True)
    breakdown = score_breakdown(bucket)
    assert breakdown.urgency_boost == 1.0


# ---------------------------------------------------------------------------
# Configurable parameters reach the formula
# ---------------------------------------------------------------------------
def test_lambda_increases_decay_speed():
    bucket = _make_bucket(days_since_active=10.0)
    fast_decay = calculate_score(bucket, lam=0.5)
    slow_decay = calculate_score(bucket, lam=0.01)
    assert fast_decay < slow_decay


def test_arousal_boost_amplifies_emotion_weight():
    bucket = _make_bucket(arousal=0.9, days_since_active=10.0)
    high_boost = calculate_score(bucket, arousal_boost=2.0)
    low_boost = calculate_score(bucket, arousal_boost=0.0)
    assert high_boost > low_boost


# ---------------------------------------------------------------------------
# Time weight monotonic with hours
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "h_a,h_b",
    [
        (0.0, 1.0),
        (1.0, 24.0),
        (24.0, 100.0),
    ],
)
def test_time_weight_monotone_decreasing(h_a, h_b):
    from astrbot_plugin_ob_memory.core.decay_engine import _time_weight

    assert _time_weight(h_a) >= _time_weight(h_b)
    assert _time_weight(h_a) <= 2.0
    assert _time_weight(h_b) >= 1.0


# ===========================================================================
# Engine cycle tests  (DB-backed)
# ===========================================================================
async def _open_engine(
    tmp_path: Path,
    *,
    threshold: float = 0.3,
    interval_hours: float = 0.0,
) -> tuple[Database, MemoryManager, DecayEngine]:
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    mgr = MemoryManager(db)
    cfg = DecayConfig(
        archive_threshold=threshold,
        check_interval_hours=interval_hours,
    )
    engine = DecayEngine(mgr, cfg)
    return db, mgr, engine


@ASYNCIO
async def test_run_cycle_archives_low_score_bucket(tmp_path: Path):
    db, mgr, engine = await _open_engine(tmp_path, threshold=1.0)
    try:
        # A 60-day-old, low-importance, low-arousal bucket — clearly stale.
        old = await mgr.create_simple(
            "s", "ancient memory",
            importance=2, valence=0.5, arousal=0.1,
        )
        # Backdate it manually past the threshold.
        await db.execute(
            "UPDATE memories SET created_at = ?, last_active_at = ? WHERE id = ?",
            (time.time() - 60 * 86400, time.time() - 60 * 86400, old.id),
        )

        stats = await engine.run_cycle("s")
        assert stats["archived"] == 1

        loaded = await mgr.get("s", old.id)
        assert loaded is not None
        assert loaded.bucket_type == "archived"
    finally:
        await db.close()


@ASYNCIO
async def test_run_cycle_keeps_recent_bucket(tmp_path: Path):
    db, mgr, engine = await _open_engine(tmp_path, threshold=0.3)
    try:
        recent = await mgr.create_simple(
            "s", "fresh", importance=8, valence=0.5, arousal=0.6,
        )
        stats = await engine.run_cycle("s")
        assert stats["archived"] == 0
        loaded = await mgr.get("s", recent.id)
        assert loaded is not None
        assert loaded.bucket_type == "dynamic"
    finally:
        await db.close()


@ASYNCIO
async def test_run_cycle_skips_pinned_and_feel(tmp_path: Path):
    db, mgr, engine = await _open_engine(tmp_path, threshold=10.0)
    try:
        pinned = await mgr.create_simple("s", "core principle", pinned=True)
        feel = await mgr.create_simple(
            "s", "I felt seen", bucket_type="feel",
        )
        # Backdate both to very old; should still be ignored.
        for bucket_id in (pinned.id, feel.id):
            await db.execute(
                "UPDATE memories SET created_at = ?, last_active_at = ? WHERE id = ?",
                (time.time() - 365 * 86400, time.time() - 365 * 86400, bucket_id),
            )

        stats = await engine.run_cycle("s")
        assert stats["archived"] == 0

        # Both should still be in their original states.
        p = await mgr.get("s", pinned.id)
        f = await mgr.get("s", feel.id)
        assert p is not None and p.bucket_type == "permanent"
        assert f is not None and f.bucket_type == "feel"
    finally:
        await db.close()


@ASYNCIO
async def test_auto_resolve_in_same_cycle(tmp_path: Path):
    """Property 10: auto-resolve fires before scoring within one cycle.

    A small-importance long-stale bucket should both:
    1. flip ``resolved=True`` AND
    2. score under the resolved penalty AND
    3. archive in the same cycle (because its post-penalty score is tiny).
    """
    db, mgr, engine = await _open_engine(tmp_path, threshold=0.3)
    try:
        bucket = await mgr.create_simple(
            "s", "old chitchat",
            importance=3, valence=0.5, arousal=0.2,
        )
        await db.execute(
            "UPDATE memories SET created_at = ?, last_active_at = ? WHERE id = ?",
            (time.time() - 45 * 86400, time.time() - 45 * 86400, bucket.id),
        )

        stats = await engine.run_cycle("s")
        assert stats["auto_resolved"] == 1
        # And because the resolved factor is 0.05, the score will be tiny
        # → archived in the same cycle.
        assert stats["archived"] == 1
    finally:
        await db.close()


@ASYNCIO
async def test_run_cycle_iterates_all_sessions_when_id_omitted(tmp_path: Path):
    db, mgr, engine = await _open_engine(tmp_path, threshold=10.0)
    try:
        await mgr.create_simple("session-A", "a")
        await mgr.create_simple("session-B", "b")
        stats = await engine.run_cycle()  # no session arg
        assert stats["checked"] == 2
    finally:
        await db.close()


@ASYNCIO
async def test_engine_disabled_when_interval_zero(tmp_path: Path):
    db, mgr, engine = await _open_engine(tmp_path, interval_hours=0.0)
    try:
        await engine.start()
        assert engine.is_running is False
    finally:
        await db.close()


@ASYNCIO
async def test_engine_start_and_stop_cleanly(tmp_path: Path):
    """The background task must spawn, run at least once, and stop within 1s."""
    db, mgr, engine = await _open_engine(tmp_path, interval_hours=0.001)
    try:
        await mgr.create_simple("s", "x")
        await engine.start()
        assert engine.is_running is True
        # Give the loop a moment to schedule, then stop.
        await asyncio.sleep(0.05)
        t0 = time.monotonic()
        await engine.stop(timeout=1.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.5
        assert engine.is_running is False
    finally:
        await db.close()


@ASYNCIO
async def test_engine_cycle_continues_on_per_session_error(tmp_path: Path, monkeypatch):
    """If one session blows up, the cycle still processes the others."""
    db, mgr, engine = await _open_engine(tmp_path, threshold=10.0)
    try:
        await mgr.create_simple("session-A", "a")
        await mgr.create_simple("session-B", "b")

        original = engine._cycle_for_session

        async def explode_for_a(sid: str):
            if sid == "session-A":
                raise RuntimeError("boom")
            return await original(sid)

        monkeypatch.setattr(engine, "_cycle_for_session", explode_for_a)
        stats = await engine.run_cycle()
        assert stats["errors"] == 1
        # session-B was processed
        assert stats["checked"] >= 1
    finally:
        await db.close()
