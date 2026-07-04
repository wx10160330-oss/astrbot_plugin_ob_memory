"""Dual-channel search: keyword fuzzy match + vector cosine similarity.

For a given query within a session we run two independent retrievers and
merge their results:

- **Keyword channel** uses ``rapidfuzz.fuzz.partial_ratio`` against the
  bucket's ``name``, ``domain[*]``, ``tags[*]`` and the first 1000 chars
  of ``content``. The four sub-scores get weights 3 / 2.5 / 2 / 1
  respectively (Ombre Brain's tuning, validated empirically).
- **Vector channel** asks the EmbeddingService for the top-N nearest
  embeddings within the session.

Buckets that came from either channel get a final blended score along
four dimensions: ``topic``, ``emotion``, ``time``, ``importance``. The
weights are configurable; defaults match Requirement 7.4.

The Resolved buckets get a ×0.3 ranking penalty (kept reachable but
ranked below unresolved alternatives) per Property 6.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Literal

try:
    from rapidfuzz import fuzz
except ImportError as _e:
    raise ImportError(
        "astrbot_plugin_ob_memory requires 'rapidfuzz'. "
        "Install it with: pip install rapidfuzz>=3.0.0"
    ) from _e

from .embedding_service import EmbeddingService
from .memory_manager import MemoryManager
from .models import MemoryBucket

logger = logging.getLogger("astrbot_plugin_ob_memory.search")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
KEYWORD_WEIGHT_NAME: float = 3.0
KEYWORD_WEIGHT_DOMAIN: float = 2.5
KEYWORD_WEIGHT_TAGS: float = 2.0
KEYWORD_WEIGHT_CONTENT: float = 1.0
"""Per-field weights inside the keyword channel (0-100 sub-scores)."""

CONTENT_PREVIEW_CHARS: int = 1000
"""How much of ``content`` participates in keyword scoring."""

DEFAULT_VECTOR_TOP_K: int = 50
DEFAULT_VECTOR_MIN_SIM: float = 0.5
"""Vector channel: pull this many candidates with cosine ≥ this threshold."""

W_TOPIC: float = 4.0
W_EMOTION: float = 2.0
W_TIME: float = 1.5
W_IMPORTANCE: float = 1.0
"""Final-score dimension weights (Requirement 7.4)."""

TIME_DECAY_RATE: float = 0.02
"""``time_score = e^(-0.02 × days)``. Slower than the bucket-recency
calculation in DecayEngine because search prefers long-tail discovery."""

RESOLVED_PENALTY: float = 0.3
"""Multiplier applied to final score for ``resolved=True`` buckets."""

VectorMatchSource = Literal["keyword", "vector", "both"]


@dataclass
class SearchHit:
    """One result of a dual-channel search.

    The ``via`` tag lets the UI annotate why a bucket showed up — useful
    in the Dashboard's Search tab and in the LLM injection block where we
    label vector-only matches with ``[语义关联]``.
    """

    bucket: MemoryBucket
    score: float
    via: VectorMatchSource
    keyword_score: float = 0.0
    vector_similarity: float = 0.0


# ---------------------------------------------------------------------------
# Sub-scoring (pure functions — easy to test in isolation)
# ---------------------------------------------------------------------------
def keyword_score(query: str, bucket: MemoryBucket) -> float:
    """Weighted fuzzy match over name/domain/tags/content.

    Returns a 0–100 normalised score. Empty query → 0 (keyword channel
    contributes nothing when there's nothing to match).
    """
    if not query:
        return 0.0

    name_score = fuzz.partial_ratio(query, bucket.name or "") * KEYWORD_WEIGHT_NAME

    if bucket.domain:
        domain_score = (
            max(fuzz.partial_ratio(query, d) for d in bucket.domain)
            * KEYWORD_WEIGHT_DOMAIN
        )
    else:
        domain_score = 0.0

    if bucket.tags:
        tag_score = (
            max(fuzz.partial_ratio(query, t) for t in bucket.tags) * KEYWORD_WEIGHT_TAGS
        )
    else:
        tag_score = 0.0

    content_score = (
        fuzz.partial_ratio(query, (bucket.content or "")[:CONTENT_PREVIEW_CHARS])
        * KEYWORD_WEIGHT_CONTENT
    )

    weight_total = (
        KEYWORD_WEIGHT_NAME
        + KEYWORD_WEIGHT_DOMAIN
        + KEYWORD_WEIGHT_TAGS
        + KEYWORD_WEIGHT_CONTENT
    )
    return (name_score + domain_score + tag_score + content_score) / weight_total


def emotion_score(
    query_valence: float | None,
    query_arousal: float | None,
    bucket: MemoryBucket,
) -> float:
    """Russell-coordinate Euclidean distance, normalised to 0-1.

    No query coordinates → neutral 0.5 (signal removed from ranking).
    """
    if query_valence is None or query_arousal is None:
        return 0.5

    qv = max(0.0, min(1.0, float(query_valence)))
    qa = max(0.0, min(1.0, float(query_arousal)))
    bv = max(0.0, min(1.0, float(bucket.valence)))
    ba = max(0.0, min(1.0, float(bucket.arousal)))
    distance = math.sqrt((qv - bv) ** 2 + (qa - ba) ** 2)
    # Maximum Euclidean distance over [0,1]^2 is sqrt(2) ≈ 1.414.
    return max(0.0, 1.0 - distance / math.sqrt(2.0))


def time_score(bucket: MemoryBucket, *, now: float | None = None) -> float:
    """``e^(-0.02 × days_since_last_active)``. Older = lower."""
    now_ts = now if now is not None else time.time()
    days_since = max(0.0, (now_ts - float(bucket.last_active_at)) / 86400.0)
    return math.exp(-TIME_DECAY_RATE * days_since)


def importance_score(bucket: MemoryBucket) -> float:
    """``importance / 10`` clamped to [0, 1]."""
    return max(1, min(10, int(bucket.importance))) / 10.0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class SearchService:
    """Dual-channel retrieval orchestrator.

    The service owns no state of its own; it just glues MemoryManager and
    EmbeddingService together. Tests construct a SearchService with a
    real MemoryManager + a stub EmbeddingService (or no embedding at all)
    to exercise both keyword-only and dual-channel paths.
    """

    def __init__(
        self,
        manager: MemoryManager,
        embedding: EmbeddingService | None = None,
    ):
        self.manager = manager
        self.embedding = embedding

    async def search(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 10,
        domain_filter: list[str] | None = None,
        query_valence: float | None = None,
        query_arousal: float | None = None,
        include_archived: bool = False,
        vector_top_k: int = DEFAULT_VECTOR_TOP_K,
        vector_min_sim: float = DEFAULT_VECTOR_MIN_SIM,
        fuzzy_threshold: float = 0.0,
    ) -> list[SearchHit]:
        """Run the dual-channel search and return ranked hits.

        Behaviour notes:
        - With an empty ``query``, keyword channel contributes 0 and
          vector channel is skipped (an empty embedding doesn't help). The
          caller should use :class:`SurfaceStrategy` instead for the
          no-query case.
        - ``include_archived=False`` filters out ``bucket_type='archived'``.
        - ``fuzzy_threshold`` filters keyword-only candidates that don't
          clear the bar; vector-channel matches always survive even if
          their keyword score is low (that's the whole point of having
          the vector channel).
        """
        if not query or not query.strip():
            return []

        # ------------------------------------------------------------------
        # 1. Pull candidates from both channels.
        # ------------------------------------------------------------------
        all_buckets = await self.manager.list_by_session(
            session_id, include_archived=include_archived
        )

        if domain_filter:
            wanted = {d.lower() for d in domain_filter if d}
            if wanted:
                all_buckets = [
                    b for b in all_buckets if any(d.lower() in wanted for d in b.domain)
                ]

        bucket_index = {b.id: b for b in all_buckets}

        # ------------------------------------------------------------------
        # 2. Vector channel (optional).
        # ------------------------------------------------------------------
        vector_hits: dict[str, float] = {}
        if self.embedding is not None and self.embedding.enabled:
            try:
                results = await self.embedding.search_similar(
                    session_id,
                    query,
                    top_k=vector_top_k,
                    min_similarity=vector_min_sim,
                )
            except Exception as e:
                logger.warning("vector channel failed: %s", e)
                results = []
            for bucket_id, sim in results:
                # Drop hits filtered out by domain pre-filter.
                if bucket_id in bucket_index:
                    vector_hits[bucket_id] = sim

        # ------------------------------------------------------------------
        # 3. Score every viable candidate.
        # ------------------------------------------------------------------
        now_ts = time.time()
        weight_sum = W_TOPIC + W_EMOTION + W_TIME + W_IMPORTANCE
        hits: list[SearchHit] = []
        for bucket in all_buckets:
            kw = keyword_score(query, bucket)
            vs = vector_hits.get(bucket.id, 0.0)

            # A bucket joins the result set if EITHER channel signals it.
            cleared_keyword = kw >= fuzzy_threshold * 100.0  # threshold is in [0,1]
            cleared_vector = bucket.id in vector_hits

            if not (cleared_keyword or cleared_vector):
                continue

            via: VectorMatchSource = (
                "both"
                if (kw > 0 and cleared_vector)
                else ("vector" if cleared_vector else "keyword")
            )

            # Normalise sub-scores to [0, 1] so the weighted sum is
            # commensurate across dimensions.
            topic = kw / 100.0
            emo = emotion_score(query_valence, query_arousal, bucket)
            tim = time_score(bucket, now=now_ts)
            imp = importance_score(bucket)

            raw = topic * W_TOPIC + emo * W_EMOTION + tim * W_TIME + imp * W_IMPORTANCE
            normalised = (raw / weight_sum) * 100.0

            if bucket.resolved:
                normalised *= RESOLVED_PENALTY

            hits.append(
                SearchHit(
                    bucket=bucket,
                    score=round(normalised, 2),
                    via=via,
                    keyword_score=round(kw, 2),
                    vector_similarity=round(vs, 4),
                )
            )

        # ------------------------------------------------------------------
        # 4. Rank.
        # ------------------------------------------------------------------
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]
