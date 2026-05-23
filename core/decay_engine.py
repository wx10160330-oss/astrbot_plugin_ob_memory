"""Memory decay engine — the "forgetting" half of the system.

The score formula is the same one Ombre Brain uses (the only material
adaptation is unit-of-time: we work in unix epoch seconds whereas Ombre
Brain works in ISO timestamps + ``datetime``). It encodes a few human
intuitions:

- Pinned / permanent / feel buckets short-circuit. They have semantic
  meaning ("core principle", "what I took away") that should not interact
  with statistical decay at all.
- Short term (≤ 3 days) memories decay slowly because they're still
  "fresh"; time weight dominates the score.
- Long term (> 3 days) memories decay according to emotional intensity:
  emotionally arousing things stay accessible longer than dull ones.
- Unresolved high-arousal memories get an urgency boost so they keep
  resurfacing until the user (or model) explicitly resolves them — this
  is the "未完结的事会被惦记" property.
- Resolved memories sink (×0.05) but stay reachable by keyword. Resolved
  + digested (a feel was written) sink even harder (×0.02).

The cycle method (:meth:`run_cycle`) does two things per session:

1. **auto-resolve** — small (importance ≤ 4) buckets that haven't been
   active for 30+ days flip to ``resolved=True`` so they don't keep
   surfacing forever
2. **archive** — buckets whose score falls below the threshold move to
   ``bucket_type='archived'`` so search no longer returns them

Both operations go through :class:`MemoryManager` so clamping invariants
apply uniformly, no naked SQL.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass

from .memory_manager import MemoryManager
from .models import MemoryBucket

logger = logging.getLogger("astrbot_plugin_ob_memory.decay")


# ---------------------------------------------------------------------------
# Tunables. Defaults match the spec; overridden via DecayConfig per plugin
# instance so users can tune via _conf_schema.json.
# ---------------------------------------------------------------------------
DEFAULT_LAMBDA: float = 0.05
"""λ — exponential decay rate per day; bigger = forgets faster."""

DEFAULT_THRESHOLD: float = 0.3
"""Activation_Score floor; below this a dynamic bucket gets archived."""

DEFAULT_INTERVAL_HOURS: float = 24.0
"""How often the background loop wakes up to scan for archive candidates."""

DEFAULT_EMOTION_BASE: float = 1.0
DEFAULT_AROUSAL_BOOST: float = 0.8
"""``emotion_weight = base + arousal * boost``."""

URGENCY_AROUSAL_THRESHOLD: float = 0.7
URGENCY_MULTIPLIER: float = 1.5
"""Unresolved buckets with ``arousal > 0.7`` get 1.5× boost."""

RESOLVED_FACTOR: float = 0.05
RESOLVED_DIGESTED_FACTOR: float = 0.02
"""Multipliers applied when ``resolved=True`` (and optionally digested)."""

SHORT_TERM_DAYS: float = 3.0
"""Boundary between short-term (time-dominant) and long-term (emotion-dominant)."""

TIME_WEIGHT_HALF_LIFE_HOURS: float = 36.0
"""Half-life of the freshness bonus (×2.0 → ×1.5 at ~36h, ×1.0 asymptote)."""

ACTIVATION_EXPONENT: float = 0.3
"""``activation_count^0.3`` — sub-linear contribution of recall count."""

PINNED_SCORE: float = 999.0
"""Sentinel score for pinned/permanent buckets so they always rank top."""

FEEL_SCORE: float = 50.0
"""Fixed score for feel buckets — they never decay or surface normally."""

AUTO_RESOLVE_IMPORTANCE_MAX: int = 4
AUTO_RESOLVE_DAYS_MIN: float = 30.0
"""Buckets with importance ≤ 4 untouched for 30+ days auto-resolve."""


# ---------------------------------------------------------------------------
# Score breakdown — used by the Dashboard's Decay Debug view and by any test
# that wants to assert per-component values.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreBreakdown:
    """Decomposition of a single Activation_Score computation.

    The fields mirror the formula step-by-step so the Dashboard can render
    them directly without re-implementing any math.

    Attributes
    ----------
    shortcut:
        ``"pinned"`` / ``"permanent"`` / ``"feel"`` if the score was
        determined by a short-circuit, otherwise ``None``.
    score:
        The final number returned by :func:`calculate_score`.
    """

    score: float
    shortcut: str | None
    importance: int
    activation_count: float
    activation_term: float
    days_since_active: float
    lambda_term: float
    time_weight: float
    emotion_weight: float
    combined_weight: float
    resolved_factor: float
    urgency_boost: float


@dataclass
class DecayConfig:
    """Snapshot of decay parameters used by one cycle.

    Built from the live plugin config, so new values picked from the
    Dashboard apply on the next cycle without restarting the engine.
    """

    decay_lambda: float = DEFAULT_LAMBDA
    archive_threshold: float = DEFAULT_THRESHOLD
    check_interval_hours: float = DEFAULT_INTERVAL_HOURS
    emotion_base: float = DEFAULT_EMOTION_BASE
    arousal_boost: float = DEFAULT_AROUSAL_BOOST


# ---------------------------------------------------------------------------
# Pure scoring function
# ---------------------------------------------------------------------------
def _time_weight(hours_since_active: float) -> float:
    """Freshness multiplier: ``1 + e^(-h/36)`` ∈ [1.0, 2.0]."""
    if hours_since_active <= 0:
        return 2.0
    return 1.0 + math.exp(-hours_since_active / TIME_WEIGHT_HALF_LIFE_HOURS)


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def calculate_score(
    bucket: MemoryBucket,
    *,
    now: float | None = None,
    lam: float = DEFAULT_LAMBDA,
    base: float = DEFAULT_EMOTION_BASE,
    arousal_boost: float = DEFAULT_AROUSAL_BOOST,
) -> float:
    """Activation_Score for a bucket at time ``now``.

    Returns a non-negative float. The components and constants come from
    the spec (see ``Requirement 5``). Short-circuits keep priority
    semantic — a pinned bucket should always rank top regardless of how
    old or rare its activation_count is.
    """
    breakdown = score_breakdown(
        bucket,
        now=now,
        lam=lam,
        base=base,
        arousal_boost=arousal_boost,
    )
    return breakdown.score


def score_breakdown(
    bucket: MemoryBucket,
    *,
    now: float | None = None,
    lam: float = DEFAULT_LAMBDA,
    base: float = DEFAULT_EMOTION_BASE,
    arousal_boost: float = DEFAULT_AROUSAL_BOOST,
) -> ScoreBreakdown:
    """Like :func:`calculate_score` but also returns every intermediate value.

    Used by the Dashboard's Decay Debug view and by property tests that
    assert specific component values.
    """
    # ------------------ Short-circuits ------------------
    if bucket.pinned or bucket.bucket_type == "permanent":
        return ScoreBreakdown(
            score=PINNED_SCORE,
            shortcut="pinned" if bucket.pinned else "permanent",
            importance=int(bucket.importance),
            activation_count=float(bucket.activation_count),
            activation_term=0.0,
            days_since_active=0.0,
            lambda_term=0.0,
            time_weight=0.0,
            emotion_weight=0.0,
            combined_weight=0.0,
            resolved_factor=1.0,
            urgency_boost=1.0,
        )
    if bucket.bucket_type == "feel":
        return ScoreBreakdown(
            score=FEEL_SCORE,
            shortcut="feel",
            importance=int(bucket.importance),
            activation_count=float(bucket.activation_count),
            activation_term=0.0,
            days_since_active=0.0,
            lambda_term=0.0,
            time_weight=0.0,
            emotion_weight=0.0,
            combined_weight=0.0,
            resolved_factor=1.0,
            urgency_boost=1.0,
        )

    now_ts = now if now is not None else time.time()
    seconds_since = max(0.0, now_ts - float(bucket.last_active_at))
    days_since = seconds_since / 86400.0
    hours_since = seconds_since / 3600.0

    # ------------------ Component values ------------------
    importance = max(1, min(10, int(bucket.importance)))
    activation = max(1.0, float(bucket.activation_count))
    activation_term = activation**ACTIVATION_EXPONENT
    lambda_term = math.exp(-lam * days_since)

    arousal = max(0.0, min(1.0, float(bucket.arousal)))
    emotion_weight = base + arousal * arousal_boost
    time_w = _time_weight(hours_since)

    # Short-term vs long-term split with a smooth crossfade around the
    # 3-day boundary. The crossfade is a half-day-wide sigmoid so:
    # - days ≪ 3: ``alpha → 1`` → time-dominant (time*0.7 + emotion*0.3)
    # - days = 3: ``alpha = 0.5`` → exact midpoint (time + emotion) / 2
    # - days ≫ 3: ``alpha → 0`` → emotion-dominant (emotion*0.7 + time*0.3)
    #
    # The midpoint is identical regardless of how it's reached, which gives
    # the formula the boundary-continuity property called out in the spec.
    alpha = _sigmoid(-(days_since - SHORT_TERM_DAYS) / 0.5)
    short = time_w * 0.7 + emotion_weight * 0.3
    long_ = emotion_weight * 0.7 + time_w * 0.3
    combined = alpha * short + (1.0 - alpha) * long_

    # ------------------ Modifiers ------------------
    if bucket.resolved and bucket.digested:
        resolved_factor = RESOLVED_DIGESTED_FACTOR
    elif bucket.resolved:
        resolved_factor = RESOLVED_FACTOR
    else:
        resolved_factor = 1.0

    urgency_boost = (
        URGENCY_MULTIPLIER
        if (arousal > URGENCY_AROUSAL_THRESHOLD and not bucket.resolved)
        else 1.0
    )

    score = (
        importance
        * activation_term
        * lambda_term
        * combined
        * resolved_factor
        * urgency_boost
    )

    return ScoreBreakdown(
        score=round(score, 4),
        shortcut=None,
        importance=importance,
        activation_count=activation,
        activation_term=activation_term,
        days_since_active=days_since,
        lambda_term=lambda_term,
        time_weight=time_w,
        emotion_weight=emotion_weight,
        combined_weight=combined,
        resolved_factor=resolved_factor,
        urgency_boost=urgency_boost,
    )


# ---------------------------------------------------------------------------
# Decay engine — periodic background scanner
# ---------------------------------------------------------------------------
class DecayEngine:
    """Background task that periodically archives stale memories.

    The engine is a thin orchestrator: scoring is a pure function above,
    storage updates go through :class:`MemoryManager`. We only own the
    asyncio task handle and the cycle bookkeeping.
    """

    def __init__(self, manager: MemoryManager, config: DecayConfig | None = None):
        self.manager = manager
        self.config = config or DecayConfig()
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    @property
    def is_running(self) -> bool:
        """``True`` once :meth:`start` has launched the background task."""
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------
    async def run_cycle(self, session_id: str | None = None) -> dict[str, int]:
        """Run one decay cycle, returning summary counters.

        If ``session_id`` is None the cycle iterates every known session.
        Per-session scoping makes this safe to invoke from the dashboard
        on demand without scanning the whole database every time.
        """
        if session_id is None:
            sessions = await self.manager.list_sessions()
        else:
            sessions = [session_id]

        totals = {"checked": 0, "auto_resolved": 0, "archived": 0, "errors": 0}
        for sid in sessions:
            try:
                stats = await self._cycle_for_session(sid)
            except Exception as e:
                logger.warning("decay cycle failed for session %s: %s", sid, e)
                totals["errors"] += 1
                continue
            totals["checked"] += stats["checked"]
            totals["auto_resolved"] += stats["auto_resolved"]
            totals["archived"] += stats["archived"]

        logger.info("decay cycle complete: %s", totals)
        return totals

    async def _cycle_for_session(self, session_id: str) -> dict[str, int]:
        """Process a single session's dynamic buckets.

        Auto-resolve fires before scoring inside the same cycle so the
        ``resolved_factor`` ×0.05 takes effect immediately (Property 10).
        """
        buckets = await self.manager.list_by_session(
            session_id,
            include_archived=False,
            bucket_types=("dynamic",),
        )
        now_ts = time.time()
        cfg = self.config
        stats = {"checked": 0, "auto_resolved": 0, "archived": 0}

        for bucket in buckets:
            if bucket.pinned or bucket.bucket_type != "dynamic":
                continue
            stats["checked"] += 1

            # ---------- 1. auto-resolve ----------
            if (
                not bucket.resolved
                and bucket.importance <= AUTO_RESOLVE_IMPORTANCE_MAX
                and ((now_ts - bucket.last_active_at) / 86400.0) > AUTO_RESOLVE_DAYS_MIN
            ):
                try:
                    updated = await self.manager.update(
                        session_id, bucket.id, resolved=True
                    )
                    if updated is not None:
                        bucket = updated  # use the freshly clamped row for scoring
                        stats["auto_resolved"] += 1
                except Exception as e:
                    logger.warning("auto-resolve failed for %s: %s", bucket.id, e)

            # ---------- 2. score and maybe archive ----------
            try:
                score = calculate_score(
                    bucket,
                    now=now_ts,
                    lam=cfg.decay_lambda,
                    base=cfg.emotion_base,
                    arousal_boost=cfg.arousal_boost,
                )
            except Exception as e:
                logger.warning("scoring failed for %s: %s", bucket.id, e)
                continue

            if score < cfg.archive_threshold:
                try:
                    if await self.manager.archive(session_id, bucket.id):
                        stats["archived"] += 1
                        logger.debug(
                            "archived %s in %s (score=%.4f < %.4f)",
                            bucket.id,
                            session_id,
                            score,
                            cfg.archive_threshold,
                        )
                except Exception as e:
                    logger.warning("archive failed for %s: %s", bucket.id, e)

        return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Spawn the background loop unless ``check_interval_hours <= 0``."""
        if self.is_running:
            return
        if self.config.check_interval_hours <= 0:
            logger.info(
                "decay engine disabled (check_interval_hours=%s)",
                self.config.check_interval_hours,
            )
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run_forever())
        logger.info(
            "decay engine started (interval=%.1fh)",
            self.config.check_interval_hours,
        )

    async def stop(self, *, timeout: float = 1.0) -> None:
        """Signal the loop to exit and wait up to ``timeout`` seconds."""
        if not self.is_running:
            return
        self._stopping.set()
        assert self._task is not None
        self._task.cancel()
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except (TimeoutError, asyncio.CancelledError):
            pass
        finally:
            self._task = None
        logger.info("decay engine stopped")

    async def _run_forever(self) -> None:
        """Background body — sleeps between cycles, cancellable mid-sleep."""
        interval_seconds = self.config.check_interval_hours * 3600.0
        while not self._stopping.is_set():
            try:
                await self.run_cycle()
            except Exception as e:
                # Property 14.3: a bad bucket / session must never stop the loop.
                logger.error("decay cycle raised; continuing: %s", e)
            try:
                # Use wait_for over a never-set event so cancel() interrupts
                # the sleep cleanly.
                await asyncio.wait_for(self._stopping.wait(), timeout=interval_seconds)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
