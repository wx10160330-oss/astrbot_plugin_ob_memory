"""End-to-end tests for ``core.memory_writer``.

Covers:
- New-bucket path (no merge candidate)
- Merge path (similarity above threshold)
- Threshold determinism (Property 5)
- Pinned write skips merge and lands in permanent
- User overrides for valence/arousal/importance beat the analyser
- ``hold_feel`` creates a feel bucket and digests the source
- ``hold_feel`` without source_bucket_id leaves no orphaned digest
- All graceful degradations: no Tagger, no Embedding, broken Embedding
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from astrbot_plugin_ob_memory.core.embedding_service import EmbeddingService
from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.core.memory_writer import MemoryWriter
from astrbot_plugin_ob_memory.core.tagger import Tagger
from astrbot_plugin_ob_memory.storage import Database, apply_migrations

ASYNCIO = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
@dataclass
class FakeLLMResponse:
    completion_text: str


class StubProvider:
    """Returns canned responses by category.

    The Tagger calls text_chat for analyse / merge / judge — we route by
    looking at the system_prompt prefix to keep the fake compact.
    """

    def __init__(
        self,
        *,
        analyse_response: str = "",
        merge_response: str = "",
    ):
        self.analyse_response = analyse_response
        self.merge_response = merge_response
        self.calls: list[tuple[str, str]] = []

    async def text_chat(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> FakeLLMResponse:
        sp = system_prompt or ""
        self.calls.append((sp[:30], prompt or ""))
        if "memory analyst" in sp.lower():
            return FakeLLMResponse(self.analyse_response)
        if "merge two related memory contents" in sp.lower():
            return FakeLLMResponse(self.merge_response)
        return FakeLLMResponse("")


class FakeEmbeddingProvider:
    """Returns vectors from a fixed mapping; unmapped strings get zeroes."""

    def __init__(self, mapping: dict[str, list[float]] | None = None, dim: int = 4):
        self._mapping = mapping or {}
        self._dim = dim
        self.calls: list[str] = []

    async def get_embedding(self, text: str) -> list[float]:
        self.calls.append(text)
        if text in self._mapping:
            return list(self._mapping[text])
        return [0.0] * self._dim

    def get_dim(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Plumbing helpers
# ---------------------------------------------------------------------------
async def _open(tmp_path: Path):
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    mgr = MemoryManager(db)
    return db, mgr


def _analyse_response(*, domain="日常", valence=0.5, arousal=0.3, name="", importance=5, tags=()) -> str:
    """Build a JSON string that Tagger.analyze would accept."""
    import json

    return json.dumps(
        {
            "domain": [domain],
            "valence": valence,
            "arousal": arousal,
            "tags": list(tags),
            "suggested_name": name,
            "importance": importance,
        }
    )


# ===========================================================================
# hold — new bucket path
# ===========================================================================
@ASYNCIO
async def test_hold_creates_new_bucket_with_analyzed_metadata(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        provider = StubProvider(
            analyse_response=_analyse_response(
                domain="求职", valence=0.8, arousal=0.7,
                name="实习offer", importance=7, tags=("实习", "offer"),
            )
        )
        writer = MemoryWriter(
            mgr,
            tagger=Tagger(context=None, fixed_provider=provider),
            embedding=None,
        )
        result = await writer.hold("s", "我今天拿到了实习 offer")
        assert result.was_merged is False
        assert result.target_bucket.name == "实习offer"
        assert result.target_bucket.domain == ["求职"]
        assert "实习" in result.target_bucket.tags
        assert result.target_bucket.importance == 7
        assert result.target_bucket.valence == pytest.approx(0.8)
    finally:
        await db.close()


@ASYNCIO
async def test_hold_user_overrides_beat_analyzer(tmp_path: Path):
    """User-supplied valence/arousal/importance must take precedence."""
    db, mgr = await _open(tmp_path)
    try:
        provider = StubProvider(
            analyse_response=_analyse_response(
                valence=0.1, arousal=0.1, importance=2,
            )
        )
        writer = MemoryWriter(
            mgr,
            tagger=Tagger(context=None, fixed_provider=provider),
            embedding=None,
        )
        result = await writer.hold(
            "s", "x",
            valence=0.9, arousal=0.8, importance=9,
        )
        assert result.target_bucket.valence == pytest.approx(0.9)
        assert result.target_bucket.arousal == pytest.approx(0.8)
        assert result.target_bucket.importance == 9
    finally:
        await db.close()


@ASYNCIO
async def test_hold_combines_user_tags_with_analyzer_tags(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        provider = StubProvider(
            analyse_response=_analyse_response(tags=("a", "b"))
        )
        writer = MemoryWriter(
            mgr,
            tagger=Tagger(context=None, fixed_provider=provider),
            embedding=None,
        )
        result = await writer.hold("s", "x", tags=["c", "a"])
        # User tags should appear first; analyser tags fill in; deduped.
        assert result.target_bucket.tags == ["c", "a", "b"]
    finally:
        await db.close()


@ASYNCIO
async def test_hold_works_without_tagger_or_embedding(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        writer = MemoryWriter(mgr, tagger=None, embedding=None)
        result = await writer.hold("s", "bare-bones write")
        assert result.was_merged is False
        loaded = await mgr.get("s", result.bucket_id)
        assert loaded is not None
        assert loaded.content == "bare-bones write"
        # Default analysis values applied.
        assert loaded.domain == ["未分类"]
        assert loaded.importance == 5
    finally:
        await db.close()


# ===========================================================================
# hold — merge path (Property 5)
# ===========================================================================
@ASYNCIO
async def test_hold_merges_when_similarity_above_threshold(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        provider = StubProvider(
            analyse_response=_analyse_response(
                domain="求职", valence=0.8, arousal=0.7,
                name="实习", importance=7, tags=("实习",),
            ),
            merge_response="合并后的统一描述",
        )
        # Both old and new content embed to the SAME vector → cosine 1.0.
        emb_provider = FakeEmbeddingProvider(
            mapping={
                "first attempt": [1.0, 0.0, 0.0, 0.0],
                "near-duplicate of the first": [1.0, 0.0, 0.0, 0.0],
            }
        )
        embedding = EmbeddingService(db, provider=emb_provider)
        tagger = Tagger(context=None, fixed_provider=provider)
        writer = MemoryWriter(mgr, tagger=tagger, embedding=embedding,
                              merge_threshold=0.85)

        first = await writer.hold("s", "first attempt")
        second = await writer.hold("s", "near-duplicate of the first")

        assert second.was_merged is True
        assert second.bucket_id == first.bucket_id
        # Merged content came from the LLM.
        loaded = await mgr.get("s", first.bucket_id)
        assert loaded is not None
        assert loaded.content == "合并后的统一描述"
    finally:
        await db.close()


@ASYNCIO
async def test_hold_creates_when_similarity_below_threshold(tmp_path: Path):
    """Property 5 (negative direction): below-threshold similarity → new bucket."""
    db, mgr = await _open(tmp_path)
    try:
        provider = StubProvider(
            analyse_response=_analyse_response(domain="日常")
        )
        emb_provider = FakeEmbeddingProvider(
            mapping={
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "totally different content": [0.0, 1.0, 0.0, 0.0],  # cos = 0
            }
        )
        embedding = EmbeddingService(db, provider=emb_provider)
        tagger = Tagger(context=None, fixed_provider=provider)
        writer = MemoryWriter(mgr, tagger=tagger, embedding=embedding,
                              merge_threshold=0.85)

        first = await writer.hold("s", "alpha")
        second = await writer.hold("s", "totally different content")
        assert second.was_merged is False
        assert second.bucket_id != first.bucket_id
    finally:
        await db.close()


@ASYNCIO
async def test_merge_skips_pinned_buckets(tmp_path: Path):
    """Pinned buckets are core principles; never merge into them."""
    db, mgr = await _open(tmp_path)
    try:
        provider = StubProvider(analyse_response=_analyse_response())
        emb_provider = FakeEmbeddingProvider(
            mapping={
                "core idea": [1.0, 0.0, 0.0, 0.0],
                "near-duplicate": [1.0, 0.0, 0.0, 0.0],
            }
        )
        embedding = EmbeddingService(db, provider=emb_provider)
        tagger = Tagger(context=None, fixed_provider=provider)
        writer = MemoryWriter(mgr, tagger=tagger, embedding=embedding)

        pinned = await writer.hold("s", "core idea", pinned=True)
        new = await writer.hold("s", "near-duplicate")
        assert new.was_merged is False
        assert new.bucket_id != pinned.bucket_id
        loaded = await mgr.get("s", pinned.bucket_id)
        assert loaded is not None
        assert loaded.pinned is True
    finally:
        await db.close()


@ASYNCIO
async def test_merge_threshold_boundary_exactly(tmp_path: Path):
    """At exactly threshold similarity (≥), merge fires."""
    db, mgr = await _open(tmp_path)
    try:
        provider = StubProvider(
            analyse_response=_analyse_response(),
            merge_response="merged",
        )
        # Hand-craft vectors so cosine ≈ 0.85 (exactly threshold).
        # With dim=2, vectors [1,0] and [0.85, 0.5268] → cos ≈ 0.85.
        # Use [0.85, sqrt(1-0.85**2)] for a unit vector.
        import math

        sim = 0.85
        emb_provider = FakeEmbeddingProvider(
            mapping={
                "first": [1.0, 0.0],
                "borderline": [sim, math.sqrt(1 - sim * sim)],
            },
            dim=2,
        )
        embedding = EmbeddingService(db, provider=emb_provider)
        tagger = Tagger(context=None, fixed_provider=provider)
        writer = MemoryWriter(mgr, tagger=tagger, embedding=embedding,
                              merge_threshold=0.85)

        first = await writer.hold("s", "first")
        second = await writer.hold("s", "borderline")
        # ≥ threshold ⇒ merge.
        assert second.was_merged is True
        assert second.bucket_id == first.bucket_id
    finally:
        await db.close()


# ===========================================================================
# hold — pinned path
# ===========================================================================
@ASYNCIO
async def test_hold_pinned_lands_in_permanent_with_importance_10(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        provider = StubProvider(
            analyse_response=_analyse_response(importance=3)  # try to lower
        )
        writer = MemoryWriter(
            mgr,
            tagger=Tagger(context=None, fixed_provider=provider),
            embedding=None,
        )
        result = await writer.hold("s", "core principle", pinned=True)
        loaded = await mgr.get("s", result.bucket_id)
        assert loaded is not None
        assert loaded.pinned is True
        assert loaded.importance == 10
        assert loaded.bucket_type == "permanent"
    finally:
        await db.close()


# ===========================================================================
# hold_feel
# ===========================================================================
@ASYNCIO
async def test_hold_feel_creates_feel_bucket(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        writer = MemoryWriter(mgr, tagger=None, embedding=None)
        result = await writer.hold_feel(
            "s", "I felt seen when she said that",
            valence=0.7,
        )
        loaded = await mgr.get("s", result.bucket_id)
        assert loaded is not None
        assert loaded.bucket_type == "feel"
        assert loaded.valence == pytest.approx(0.7)
        assert loaded.model_valence == pytest.approx(0.7)
        assert result.source_marked_digested is False
    finally:
        await db.close()


@ASYNCIO
async def test_hold_feel_marks_source_as_digested(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        writer = MemoryWriter(mgr, tagger=None, embedding=None)
        # First, store a normal event.
        event = await writer.hold("s", "she got into a fight with him")
        assert event.target_bucket.digested is False

        # Then digest it via a feel.
        feel = await writer.hold_feel(
            "s", "I see her growing through this",
            source_bucket_id=event.bucket_id,
            valence=0.6,
        )
        assert feel.source_marked_digested is True

        loaded_event = await mgr.get("s", event.bucket_id)
        assert loaded_event is not None
        assert loaded_event.digested is True
        assert loaded_event.model_valence == pytest.approx(0.6)
    finally:
        await db.close()


@ASYNCIO
async def test_hold_feel_handles_missing_source_gracefully(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        writer = MemoryWriter(mgr, tagger=None, embedding=None)
        result = await writer.hold_feel(
            "s", "reflection",
            source_bucket_id="nonexistent-id-xxx",
            valence=0.5,
        )
        # The feel bucket is still created; only the digest mark fails.
        assert result.source_marked_digested is False
        assert await mgr.get("s", result.bucket_id) is not None
    finally:
        await db.close()


@ASYNCIO
async def test_hold_feel_does_not_use_tagger(tmp_path: Path):
    """Feel writes must NOT call the analyser — feel content is its own analysis."""
    db, mgr = await _open(tmp_path)
    try:
        provider = StubProvider(analyse_response=_analyse_response())
        writer = MemoryWriter(
            mgr,
            tagger=Tagger(context=None, fixed_provider=provider),
            embedding=None,
        )
        await writer.hold_feel("s", "x", valence=0.5)
        # Stub provider was never asked anything.
        assert provider.calls == []
    finally:
        await db.close()


@ASYNCIO
async def test_hold_feel_blank_source_string_is_treated_as_none(tmp_path: Path):
    """``source_bucket_id=""`` must NOT trigger a 'mark as digested' attempt."""
    db, mgr = await _open(tmp_path)
    try:
        writer = MemoryWriter(mgr, tagger=None, embedding=None)
        result = await writer.hold_feel("s", "x", source_bucket_id="   ")
        assert result.source_marked_digested is False
    finally:
        await db.close()


# ===========================================================================
# Robustness: invalid inputs
# ===========================================================================
@ASYNCIO
async def test_hold_rejects_empty_content(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        writer = MemoryWriter(mgr, tagger=None, embedding=None)
        with pytest.raises(ValueError):
            await writer.hold("s", "")
        with pytest.raises(ValueError):
            await writer.hold("s", "   ")
    finally:
        await db.close()


@ASYNCIO
async def test_hold_feel_rejects_empty_content(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        writer = MemoryWriter(mgr, tagger=None, embedding=None)
        with pytest.raises(ValueError):
            await writer.hold_feel("s", "")
    finally:
        await db.close()


@ASYNCIO
async def test_hold_continues_when_embedding_provider_breaks(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        class Boom:
            async def get_embedding(self, text):
                raise RuntimeError("provider down")

            def get_dim(self):
                return 0

        embedding = EmbeddingService(db, provider=Boom())
        writer = MemoryWriter(mgr, tagger=None, embedding=embedding)
        # Must not raise — embedding errors swallowed.
        result = await writer.hold("s", "content")
        assert await mgr.get("s", result.bucket_id) is not None
    finally:
        await db.close()
