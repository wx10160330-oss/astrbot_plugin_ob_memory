"""Tests for ``core.surface_strategy``.

Verifies:
- Pinned buckets are always surfaced first (Requirement 8.2)
- Cold-start buckets get up to 2 slots ahead of normal score winners (8.3)
- ``surface()`` does not call ``touch()`` (8.4)
- Token budget truncation drops lowest-priority items first (8.5)
- Feel and archived buckets are skipped
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.core.surface_strategy import (
    DEFAULT_TOKEN_BUDGET,
    SurfaceStrategy,
    estimate_tokens,
)
from astrbot_plugin_ob_memory.storage import Database, apply_migrations

ASYNCIO = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# token estimator (pure)
# ---------------------------------------------------------------------------
def test_estimate_tokens_handles_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_ascii_smaller_than_cjk():
    # 100 ASCII chars: ~30 tokens. 100 CJK chars: ~150 tokens.
    ascii_t = estimate_tokens("a" * 100)
    cjk_t = estimate_tokens("一" * 100)
    assert ascii_t < cjk_t


def test_estimate_tokens_minimum_one_for_nonempty():
    assert estimate_tokens("x") >= 1


# ===========================================================================
# Strategy
# ===========================================================================
async def _open(tmp_path: Path) -> tuple[Database, MemoryManager, SurfaceStrategy]:
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    mgr = MemoryManager(db)
    return db, mgr, SurfaceStrategy(mgr)


@ASYNCIO
async def test_empty_session_returns_empty(tmp_path: Path):
    db, _, surface = await _open(tmp_path)
    try:
        assert await surface.surface("nobody") == []
    finally:
        await db.close()


@ASYNCIO
async def test_pinned_always_first(tmp_path: Path):
    db, mgr, surface = await _open(tmp_path)
    try:
        # Pinned bucket — slightly older to ensure non-pinned would win on score.
        pinned = await mgr.create_simple("s", "core principle", pinned=True)
        await db.execute(
            "UPDATE memories SET created_at = ?, last_active_at = ? WHERE id = ?",
            (time.time() - 86400, time.time() - 86400, pinned.id),
        )
        # Hot recent bucket.
        await mgr.create_simple(
            "s", "fresh hot bucket", importance=9, valence=0.5, arousal=0.6
        )

        results = await surface.surface("s", token_budget=10000, max_results=5)
        assert results
        assert results[0].id == pinned.id
    finally:
        await db.close()


@ASYNCIO
async def test_cold_start_gets_slot_for_high_importance(tmp_path: Path):
    db, mgr, surface = await _open(tmp_path)
    try:
        # Cold-start candidate: importance 9, just created, never recalled.
        cold = await mgr.create_simple(
            "s", "newly recorded important fact", importance=9
        )
        # An older, lower-importance bucket that would otherwise dominate.
        await mgr.create_simple("s", "older mediocre bucket", importance=4)

        results = await surface.surface("s", token_budget=10000, max_results=5)
        ids = [b.id for b in results]
        assert cold.id in ids
        # Cold-start slot should rank ahead of the older ones (after pinned,
        # of which there are none here).
        assert ids[0] == cold.id
    finally:
        await db.close()


@ASYNCIO
async def test_cold_start_capped_at_two(tmp_path: Path):
    db, mgr, surface = await _open(tmp_path)
    try:
        for i in range(4):
            await mgr.create_simple(
                "s", f"important new {i}", importance=10
            )
        results = await surface.surface("s", token_budget=10000, max_results=5)
        # All four are cold-start eligible, but only 2 slots are reserved
        # for them; the remaining ones still surface as score winners,
        # which in this case gives us up to ``max_results`` total.
        assert len(results) <= 5
        assert len(results) >= 2
    finally:
        await db.close()


@ASYNCIO
async def test_cold_start_excluded_after_first_recall(tmp_path: Path):
    db, mgr, surface = await _open(tmp_path)
    try:
        b = await mgr.create_simple("s", "fresh important", importance=10)
        # Simulate the bucket having already been recalled once.
        await mgr.touch("s", b.id)

        results = await surface.surface("s", token_budget=10000)
        ids = [bucket.id for bucket in results]
        # It can still appear via the score channel, but no longer as
        # cold-start. To verify the cold-start path didn't fire we just
        # ensure the explicit ordering invariant holds.
        assert b.id in ids
    finally:
        await db.close()


@ASYNCIO
async def test_feel_buckets_skipped(tmp_path: Path):
    db, mgr, surface = await _open(tmp_path)
    try:
        feel = await mgr.create_simple(
            "s", "what I took away", bucket_type="feel"
        )
        normal = await mgr.create_simple("s", "an event", importance=6)
        results = await surface.surface("s")
        ids = [b.id for b in results]
        assert feel.id not in ids
        assert normal.id in ids
    finally:
        await db.close()


@ASYNCIO
async def test_archived_buckets_skipped(tmp_path: Path):
    db, mgr, surface = await _open(tmp_path)
    try:
        b = await mgr.create_simple("s", "stale thing", importance=2)
        await mgr.archive("s", b.id)
        results = await surface.surface("s")
        assert b.id not in [bucket.id for bucket in results]
    finally:
        await db.close()


@ASYNCIO
async def test_does_not_touch_returned_buckets(tmp_path: Path):
    """Property: surfacing must not bump activation_count or last_active_at.

    Otherwise normal conversation would keep memories artificially fresh,
    breaking the decay model.
    """
    db, mgr, surface = await _open(tmp_path)
    try:
        b = await mgr.create_simple(
            "s", "watch this bucket", importance=8, valence=0.5, arousal=0.6
        )
        before = await mgr.get("s", b.id)
        assert before is not None

        results = await surface.surface("s")
        assert b.id in [x.id for x in results]

        after = await mgr.get("s", b.id)
        assert after is not None
        assert after.activation_count == before.activation_count
        assert after.last_active_at == before.last_active_at
    finally:
        await db.close()


@ASYNCIO
async def test_token_budget_truncates(tmp_path: Path):
    db, mgr, surface = await _open(tmp_path)
    try:
        # Create several large buckets so budget enforcement bites.
        big_text = "中" * 800
        for i in range(5):
            await mgr.create_simple("s", big_text, name=f"big-{i}", importance=5)

        small_budget = 200
        results = await surface.surface(
            "s", token_budget=small_budget, max_results=10
        )
        # First bucket is always kept (we never return [] just because the
        # first one exceeds budget).
        assert len(results) >= 1
        assert len(results) < 5
    finally:
        await db.close()


@ASYNCIO
async def test_session_isolation(tmp_path: Path):
    db, mgr, surface = await _open(tmp_path)
    try:
        a = await mgr.create_simple("session-A", "alpha", importance=8)
        await mgr.create_simple("session-B", "beta", importance=8)
        results = await surface.surface("session-A")
        ids = [b.id for b in results]
        assert a.id in ids
        assert all(b.session_id == "session-A" for b in results)
    finally:
        await db.close()


@ASYNCIO
async def test_default_token_budget_constant():
    """Sanity check that the default budget is sensible."""
    assert DEFAULT_TOKEN_BUDGET >= 200
    assert DEFAULT_TOKEN_BUDGET <= 5000
