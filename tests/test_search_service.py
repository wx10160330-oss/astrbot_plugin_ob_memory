"""Unit + integration tests for ``core.search_service``.

Covers:
- Per-field keyword weighting (name × 3, domain × 2.5, tags × 2, content × 1)
- Emotion / time / importance sub-scores
- Resolved penalty (Property 6)
- Embedding fallback when provider unavailable (Property 8)
- Domain pre-filter
- Vector channel only matches still surface
- Result limit and session isolation
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from astrbot_plugin_ob_memory.core.embedding_service import EmbeddingService
from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.core.models import MemoryBucket
from astrbot_plugin_ob_memory.core.search_service import (
    SearchService,
    emotion_score,
    importance_score,
    keyword_score,
    time_score,
)
from astrbot_plugin_ob_memory.storage import Database, apply_migrations

ASYNCIO = pytest.mark.asyncio


def _bucket(
    *,
    bid: str = "abc",
    name: str = "",
    content: str = "",
    domain: list[str] | None = None,
    tags: list[str] | None = None,
    valence: float = 0.5,
    arousal: float = 0.3,
    importance: int = 5,
    resolved: bool = False,
    last_active_at: float | None = None,
) -> MemoryBucket:
    now = last_active_at if last_active_at is not None else time.time()
    return MemoryBucket(
        id=bid,
        session_id="s",
        content=content,
        name=name,
        domain=list(domain or []),
        tags=list(tags or []),
        valence=valence,
        arousal=arousal,
        importance=importance,
        resolved=resolved,
        created_at=now,
        last_active_at=now,
    )


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------
def test_keyword_zero_for_empty_query():
    assert keyword_score("", _bucket(name="x")) == 0.0


def test_keyword_name_weight_dominates_content():
    """A name-only match should outscore a content-only match of equal exactness."""
    name_match = _bucket(name="实习offer", content="无关内容")
    content_match = _bucket(name="无关名字", content="实习offer 无关")
    assert keyword_score("实习offer", name_match) > keyword_score(
        "实习offer", content_match
    )


def test_keyword_tag_match_above_threshold():
    tagged = _bucket(name="无关", content="无关", tags=["实习", "offer"])
    sc = keyword_score("实习", tagged)
    assert sc > 0


def test_keyword_domain_match_above_threshold():
    bucket = _bucket(name="无关", content="无关", domain=["求职", "成长"])
    sc = keyword_score("求职", bucket)
    assert sc > 0


# ---------------------------------------------------------------------------
# Emotion / time / importance
# ---------------------------------------------------------------------------
def test_emotion_score_neutral_when_query_missing():
    bucket = _bucket(valence=0.9, arousal=0.1)
    assert emotion_score(None, None, bucket) == 0.5


def test_emotion_score_perfect_match():
    bucket = _bucket(valence=0.7, arousal=0.4)
    assert emotion_score(0.7, 0.4, bucket) == pytest.approx(1.0, abs=1e-6)


def test_emotion_score_diagonal_extremes():
    # Query at (0,0); bucket at (1,1) → distance sqrt(2), normalised → 0.
    bucket = _bucket(valence=1.0, arousal=1.0)
    assert emotion_score(0.0, 0.0, bucket) == pytest.approx(0.0, abs=1e-6)


def test_time_score_decreases_with_age():
    now = time.time()
    fresh = _bucket(last_active_at=now)
    old = _bucket(last_active_at=now - 30 * 86400)
    assert time_score(fresh, now=now) > time_score(old, now=now)


def test_importance_score_normalised():
    assert importance_score(_bucket(importance=10)) == 1.0
    assert importance_score(_bucket(importance=1)) == 0.1
    assert importance_score(_bucket(importance=99)) == 1.0  # clamped


# ===========================================================================
# Service-level (DB-backed) tests
# ===========================================================================
class FakeEmbeddingProvider:
    def __init__(self, mapping: dict[str, list[float]] | None = None, dim: int = 4):
        self._mapping = mapping or {}
        self._dim = dim

    async def get_embedding(self, text: str) -> list[float]:
        if text in self._mapping:
            return list(self._mapping[text])
        return [0.0] * self._dim

    def get_dim(self) -> int:
        return self._dim


async def _open(tmp_path: Path) -> tuple[Database, MemoryManager]:
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    return db, MemoryManager(db)


@ASYNCIO
async def test_keyword_only_path_when_no_embedding(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        a = await mgr.create_simple("s", "我今天拿到实习 offer", name="实习offer 获得")
        await mgr.create_simple("s", "完全无关的内容", name="完全无关")
        svc = SearchService(mgr, embedding=None)
        hits = await svc.search("s", "实习", limit=10)
        assert len(hits) >= 1
        assert hits[0].bucket.id == a.id
        assert hits[0].via == "keyword"
    finally:
        await db.close()


@ASYNCIO
async def test_vector_channel_surfaces_low_keyword_match(tmp_path: Path):
    """Property 8 (positive direction): vector match brings up a bucket
    even when keyword score alone wouldn't survive the threshold."""
    db, mgr = await _open(tmp_path)
    try:
        # Bucket has zero overlap with the query string.
        target = await mgr.create_simple(
            "s", "completely unrelated sentence", name="totally different"
        )
        provider = FakeEmbeddingProvider(
            mapping={
                "completely unrelated sentence": [1.0, 0.0, 0.0, 0.0],
                "另一个语义近似的查询": [1.0, 0.0, 0.0, 0.0],
            }
        )
        embedding = EmbeddingService(db, provider=provider)
        await embedding.generate_and_store(target.id, "completely unrelated sentence")

        svc = SearchService(mgr, embedding=embedding)
        # Use a fuzzy_threshold high enough to disqualify the keyword path.
        hits = await svc.search(
            "s", "另一个语义近似的查询", fuzzy_threshold=0.9, limit=5
        )
        ids = [h.bucket.id for h in hits]
        assert target.id in ids
        # And it must be tagged as a vector match.
        via = next(h.via for h in hits if h.bucket.id == target.id)
        assert via in ("vector", "both")
    finally:
        await db.close()


@ASYNCIO
async def test_resolved_penalty_keeps_bucket_reachable(tmp_path: Path):
    """Property 6: resolved buckets stay reachable but rank below unresolved."""
    db, mgr = await _open(tmp_path)
    try:
        live = await mgr.create_simple(
            "s", "实习经历记录", name="实习经历", importance=5
        )
        old = await mgr.create_simple(
            "s", "另一份实习经历", name="实习经历2", importance=5
        )
        # Resolve the second one.
        await mgr.update("s", old.id, resolved=True)

        svc = SearchService(mgr, embedding=None)
        hits = await svc.search("s", "实习经历", limit=10)
        ids = [h.bucket.id for h in hits]

        # Both should be in the result set.
        assert live.id in ids
        assert old.id in ids
        # Live bucket ranks above the resolved one.
        live_rank = ids.index(live.id)
        old_rank = ids.index(old.id)
        assert live_rank < old_rank
        # Resolved bucket score is reduced by the 0.3 penalty.
        live_score = next(h.score for h in hits if h.bucket.id == live.id)
        old_score = next(h.score for h in hits if h.bucket.id == old.id)
        assert old_score < live_score * 0.5  # comfortably reduced
    finally:
        await db.close()


@ASYNCIO
async def test_embedding_failure_falls_back_to_keyword(tmp_path: Path):
    """Property 8 (negative direction): a broken provider must not break search."""
    db, mgr = await _open(tmp_path)
    try:
        await mgr.create_simple("s", "keyword target", name="alpha")

        class Boom:
            async def get_embedding(self, text):
                raise RuntimeError("provider down")

            def get_dim(self):
                return 0

        embedding = EmbeddingService(db, provider=Boom())
        svc = SearchService(mgr, embedding=embedding)
        hits = await svc.search("s", "alpha", limit=5)
        # Keyword channel still works — we got a hit.
        assert len(hits) >= 1
        assert hits[0].bucket.name == "alpha"
    finally:
        await db.close()


@ASYNCIO
async def test_domain_filter_excludes_unmatched(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        await mgr.create_simple(
            "s", "求职相关的事", name="求职", domain=["求职"]
        )
        await mgr.create_simple(
            "s", "无关日常", name="日常", domain=["日常"]
        )

        svc = SearchService(mgr, embedding=None)
        hits = await svc.search("s", "事", domain_filter=["求职"], limit=5)
        names = {h.bucket.name for h in hits}
        assert "求职" in names
        assert "日常" not in names
    finally:
        await db.close()


@ASYNCIO
async def test_archived_excluded_by_default(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        target = await mgr.create_simple("s", "存档候选", name="存档")
        await mgr.archive("s", target.id)

        svc = SearchService(mgr, embedding=None)
        hits = await svc.search("s", "存档")
        assert hits == []

        # Explicit include returns it.
        hits_with = await svc.search("s", "存档", include_archived=True)
        assert any(h.bucket.id == target.id for h in hits_with)
    finally:
        await db.close()


@ASYNCIO
async def test_empty_query_returns_empty_list(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        await mgr.create_simple("s", "x")
        svc = SearchService(mgr, embedding=None)
        assert await svc.search("s", "") == []
        assert await svc.search("s", "   ") == []
    finally:
        await db.close()


@ASYNCIO
async def test_limit_respected(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        for i in range(15):
            await mgr.create_simple("s", f"keyword-{i}", name=f"hit-{i}")
        svc = SearchService(mgr, embedding=None)
        hits = await svc.search("s", "keyword", limit=5)
        assert len(hits) == 5
    finally:
        await db.close()


@ASYNCIO
async def test_session_isolation(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        await mgr.create_simple("session-A", "shared keyword", name="alpha")
        await mgr.create_simple("session-B", "shared keyword", name="beta")
        svc = SearchService(mgr, embedding=None)
        hits = await svc.search("session-A", "shared")
        names = {h.bucket.name for h in hits}
        assert names == {"alpha"}
    finally:
        await db.close()
