"""Active surfacing — the "what's on my mind" memory pool.

Where :class:`SearchService` answers "find me things matching this query",
:class:`SurfaceStrategy` answers "what should naturally come up right now,
without me asking?". This is the channel the on_llm_request hook uses
when the user's prompt doesn't have an obvious search target — the AI
silently brings up unresolved or important things from past conversations
so it can refer to them naturally.

Selection priorities (high to low):

1. **Pinned** ("core principle") buckets — always shown, regardless of
   score. There are typically only a handful per session.
2. **Recent** buckets — the most recently created N buckets (default 1),
   regardless of importance or activation. Guarantees that what just
   happened a moment ago can be brought to mind even when the session
   pool contains hundreds of older high-weight memories. Particularly
   useful when many windows share one pool (``user`` scope_mode or
   ``unify_groups_into_user``) so the newest item never gets crowded
   out by old heavy hitters.
3. **Cold-start** buckets — freshly stored (created within last 24h),
   high-importance (≥ 8), not yet retrieved (activation_count == 0).
   These are memories the model just decided were important enough to
   record; surfacing them once ensures they land in the next conversation
   instead of waiting to be searched. Up to 2 per surface call.
4. **Activation_Score winners** — by descending DecayEngine score. This
   naturally promotes high-arousal unresolved memories thanks to the
   urgency boost in the decay formula.

The strategy MUST NOT call ``MemoryManager.touch`` on its picks.
Surfacing is "browsing" — it should not reset the decay timer or bump
activation_count, otherwise normal conversation would keep memories
artificially fresh forever.

Token budgeting is a coarse approximation: we treat each Chinese char as
1.5 tokens and each ASCII char as 0.3 (Ombre Brain's heuristic). The
budget is enforced by sequentially adding sorted hits and stopping when
the projected total tokens exceeds the budget.
"""

from __future__ import annotations

import logging
import time

from .decay_engine import DecayConfig, calculate_score
from .memory_manager import MemoryManager
from .models import MemoryBucket

logger = logging.getLogger("astrbot_plugin_ob_memory.surface")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
COLD_START_IMPORTANCE_MIN: int = 8
COLD_START_AGE_SECONDS: float = 24 * 3600.0
COLD_START_MAX: int = 2
"""Conditions for a bucket to count as a cold-start candidate."""

DEFAULT_RECENT_COUNT: int = 1
"""How many of the most-recently-created buckets always surface.

A dedicated lane that sidesteps score-based competition. Set to 0 to
disable. Increasing the value helps shared pools (``user`` scope or
``unify_groups_into_user``) where new memories would otherwise be
drowned out by older heavy-weight memories.
"""

DEFAULT_TOKEN_BUDGET: int = 800
DEFAULT_MAX_RESULTS: int = 5


# ---------------------------------------------------------------------------
# Lightweight token estimation (Ombre Brain heuristic)
# ---------------------------------------------------------------------------
def _is_cjk(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    # CJK Unified, Hiragana, Katakana, full-width punct
    return (
        0x3000 <= code <= 0x9FFF
        or 0xFF00 <= code <= 0xFFEF
        or 0x20000 <= code <= 0x2FFFF
    )


def estimate_tokens(text: str) -> int:
    """Rough token count: CJK char ~1.5, ASCII char ~0.3.

    Good enough for budget enforcement; we never feed this number to a
    real tokeniser. It overestimates short ASCII texts and slightly
    underestimates long Chinese ones, both error directions are safe
    (we never blow the budget; we may end up under-using it).
    """
    if not text:
        return 0
    cjk = sum(1 for c in text if _is_cjk(c))
    other = len(text) - cjk
    return int(cjk * 1.5 + other * 0.3) + 1  # +1 to never report 0 for non-empty


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class SurfaceStrategy:
    """Picks memories to push into the LLM context proactively."""

    def __init__(
        self,
        manager: MemoryManager,
        decay_config: DecayConfig | None = None,
    ):
        self.manager = manager
        self.decay_config = decay_config or DecayConfig()

    async def surface(
        self,
        session_id: str,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        max_results: int = DEFAULT_MAX_RESULTS,
        recent_count: int = DEFAULT_RECENT_COUNT,
    ) -> list[MemoryBucket]:
        """Return the buckets that should be brought to mind right now.

        The result is ordered by priority — caller can render them in
        order. Token budget is enforced *after* sorting so the highest-
        priority items always make it in.

        ``recent_count`` reserves N slots for the most recently created
        non-pinned buckets so brand-new memories don't get crowded out
        by older heavy hitters. Set to 0 to disable (legacy behaviour).
        """
        all_buckets = await self.manager.list_by_session(
            session_id, include_archived=False
        )
        if not all_buckets:
            return []

        now_ts = time.time()
        cfg = self.decay_config

        pinned: list[MemoryBucket] = []
        eligible: list[MemoryBucket] = []  # not pinned, not feel/archived
        cold_start: list[MemoryBucket] = []
        candidates: list[tuple[float, MemoryBucket]] = []

        for b in all_buckets:
            if b.pinned or b.bucket_type == "permanent":
                pinned.append(b)
                continue
            if b.bucket_type == "feel":
                # Feel buckets never participate in passive surfacing.
                continue
            if b.bucket_type == "archived":
                continue

            eligible.append(b)

            # Cold-start detection: high-importance bucket created very
            # recently and not yet recalled. activation_count == 0 is
            # required so an already-recalled bucket doesn't qualify.
            age_seconds = now_ts - float(b.created_at)
            if (
                b.importance >= COLD_START_IMPORTANCE_MIN
                and age_seconds <= COLD_START_AGE_SECONDS
                and float(b.activation_count) == 0.0
                and not b.resolved
            ):
                cold_start.append(b)
                continue

            # Otherwise: rank by Activation_Score descending.
            score = calculate_score(
                b,
                now=now_ts,
                lam=cfg.decay_lambda,
                base=cfg.emotion_base,
                arousal_boost=cfg.arousal_boost,
            )
            candidates.append((score, b))

        # Recent lane: newest non-pinned buckets, capped at recent_count.
        # This sidesteps score-based competition so a freshly stored
        # memory always gets a moment in the spotlight even when the
        # session pool has hundreds of older heavy-weight memories
        # (most common after enabling ``unify_groups_into_user`` or
        # ``scope_mode = user`` with lots of accumulated history).
        recent: list[MemoryBucket] = []
        rc = max(0, int(recent_count))
        if rc > 0 and eligible:
            recent = sorted(eligible, key=lambda b: -float(b.created_at))[:rc]

        # Cold-start: prefer the most important ones first; cap at 2.
        cold_start.sort(key=lambda b: (-b.importance, -b.created_at))
        cold_start = cold_start[:COLD_START_MAX]

        # Score winners: top-scoring among the rest, after cold-start cap.
        candidates.sort(key=lambda t: t[0], reverse=True)

        # Compose final ordering:
        #   pinned (all) → recent (≤recent_count) → cold_start (≤2)
        #   → candidates → truncate
        ordered: list[MemoryBucket] = []
        seen: set[str] = set()
        for source in (pinned, recent, cold_start, [b for _, b in candidates]):
            for bucket in source:
                if bucket.id in seen:
                    continue
                seen.add(bucket.id)
                ordered.append(bucket)

        # Apply max_results cap first (cheap), then token budget (precise).
        ordered = ordered[:max_results]

        if token_budget > 0:
            running = 0
            kept: list[MemoryBucket] = []
            for bucket in ordered:
                projected = (
                    running
                    + estimate_tokens(bucket.content)
                    + estimate_tokens(bucket.name)
                )
                if kept and projected > token_budget:
                    break
                kept.append(bucket)
                running = projected
            ordered = kept

        return ordered
