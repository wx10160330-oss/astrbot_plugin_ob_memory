"""Tests for the additional LLM tools: ``record_diary`` and
``reflect_memory``, plus the enhanced ``recall_memory`` modes
(``domain="feel"`` channel and ``importance_min`` batch fetch).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from astrbot_plugin_ob_memory.core.embedding_service import EmbeddingService
from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.core.memory_writer import MemoryWriter
from astrbot_plugin_ob_memory.core.search_service import SearchService
from astrbot_plugin_ob_memory.core.tagger import Tagger
from astrbot_plugin_ob_memory.handlers.llm_tools import MemoryToolsMixin
from astrbot_plugin_ob_memory.storage import Database, apply_migrations

ASYNCIO = pytest.mark.asyncio


def _bind(method, instance):
    """Call a mixin method as if bound to ``instance``."""
    return method.__get__(instance, instance.__class__)


@dataclass
class FakeEvent:
    unified_msg_origin: str = "qq:GroupMessage:99"


class FakeAssembled(MemoryToolsMixin):
    """Inherits from the mixin so internal helper methods like
    ``_reflect_connection_hint`` resolve correctly."""

    def __init__(self):
        self.manager: MemoryManager | None = None
        self.writer: MemoryWriter | None = None
        self.search: SearchService | None = None
        self.embedding: EmbeddingService | None = None


@dataclass
class FakeLLMResponse:
    completion_text: str


class StubLLM:
    """Returns canned responses based on which prompt-template is sent."""

    def __init__(self, *, analyze: str = "", digest: str = "", merge: str = ""):
        self.analyze = analyze
        self.digest = digest
        self.merge = merge

    async def text_chat(
        self, prompt: str | None = None, system_prompt: str | None = None, **kw
    ) -> FakeLLMResponse:
        sp = (system_prompt or "").lower()
        if "memory analyst" in sp:
            return FakeLLMResponse(self.analyze)
        if "split a long diary-like passage" in sp:
            return FakeLLMResponse(self.digest)
        if "merge two related memory contents" in sp:
            return FakeLLMResponse(self.merge)
        return FakeLLMResponse("")


def _analyze_payload(**kw) -> str:
    payload = {
        "domain": kw.get("domain", ["UNCAT"]),
        "valence": kw.get("valence", 0.5),
        "arousal": kw.get("arousal", 0.3),
        "tags": kw.get("tags", []),
        "suggested_name": kw.get("name", ""),
        "importance": kw.get("importance", 5),
    }
    return json.dumps(payload)


def _digest_payload(entries: list[dict]) -> str:
    return json.dumps(entries)


async def _open_stack(
    tmp_path: Path, *, llm: StubLLM | None = None
) -> tuple[Database, FakeAssembled]:
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    obj = FakeAssembled()
    obj.manager = MemoryManager(db)
    provider = llm or StubLLM(analyze=_analyze_payload())
    obj.writer = MemoryWriter(
        obj.manager,
        tagger=Tagger(context=None, fixed_provider=provider),
        embedding=None,
    )
    obj.search = SearchService(obj.manager, embedding=None)
    return db, obj


SID = "qq:GroupMessage:99"


# ===========================================================================
# record_diary
# ===========================================================================
@ASYNCIO
async def test_record_diary_short_input_takes_fast_path(tmp_path: Path):
    """Inputs shorter than 30 chars do not trigger the digest LLM call."""
    llm = StubLLM(analyze=_analyze_payload())
    db, obj = await _open_stack(tmp_path, llm=llm)
    try:
        method = _bind(MemoryToolsMixin.record_diary, obj)
        result = await method(FakeEvent(), "short input")
        assert "1" in result
        buckets = await obj.manager.list_by_session(SID)
        assert len(buckets) == 1
    finally:
        await db.close()


@ASYNCIO
async def test_record_diary_splits_long_passage(tmp_path: Path):
    """A long input goes through the digest LLM call and creates multiple buckets."""
    digest_entries = [
        {
            "name": "entry-1",
            "content": "first event content here",
            "domain": ["d1"],
            "valence": 0.6,
            "arousal": 0.3,
            "tags": ["t1"],
            "importance": 5,
        },
        {
            "name": "entry-2",
            "content": "second event content here",
            "domain": ["d2"],
            "valence": 0.7,
            "arousal": 0.4,
            "tags": ["t2"],
            "importance": 4,
        },
        {
            "name": "entry-3",
            "content": "third event content here",
            "domain": ["d3"],
            "valence": 0.3,
            "arousal": 0.5,
            "tags": ["t3"],
            "importance": 6,
        },
    ]
    llm = StubLLM(
        analyze=_analyze_payload(),
        digest=_digest_payload(digest_entries),
    )
    db, obj = await _open_stack(tmp_path, llm=llm)
    try:
        long_text = "x" * 200  # well above 30-char fast-path threshold
        method = _bind(MemoryToolsMixin.record_diary, obj)
        result = await method(FakeEvent(), long_text)
        assert "3" in result
        buckets = await obj.manager.list_by_session(SID)
        assert len(buckets) == 3
        names = {b.name for b in buckets}
        assert "entry-1" in names
        assert "entry-2" in names
        assert "entry-3" in names
    finally:
        await db.close()


@ASYNCIO
async def test_record_diary_empty_input(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        method = _bind(MemoryToolsMixin.record_diary, obj)
        result = await method(FakeEvent(), "")
        # Returns Chinese string; just verify nothing was created.
        buckets = await obj.manager.list_by_session(SID)
        assert len(buckets) == 0
        assert isinstance(result, str)
    finally:
        await db.close()


@ASYNCIO
async def test_record_diary_digest_failure_falls_back(tmp_path: Path):
    """When digest returns malformed output, fall back to single-bucket store."""
    llm = StubLLM(
        analyze=_analyze_payload(),
        digest="not json at all",
    )
    db, obj = await _open_stack(tmp_path, llm=llm)
    try:
        long_text = "x" * 200
        method = _bind(MemoryToolsMixin.record_diary, obj)
        result = await method(FakeEvent(), long_text)
        assert isinstance(result, str)
        buckets = await obj.manager.list_by_session(SID)
        assert len(buckets) == 1
    finally:
        await db.close()


# ===========================================================================
# recall_memory enhanced modes
# ===========================================================================
@ASYNCIO
async def test_recall_memory_feel_channel(tmp_path: Path):
    """domain='feel' returns feel-type buckets."""
    db, obj = await _open_stack(tmp_path)
    try:
        await obj.writer.hold(SID, "event A content")
        await obj.writer.hold_feel(SID, "feel one - growth observation")
        await obj.writer.hold_feel(SID, "feel two - calm observation")

        method = _bind(MemoryToolsMixin.recall_memory, obj)
        result = await method(FakeEvent(), domain="feel")
        # Both feels should appear, the event should not
        assert "growth observation" in result or "calm observation" in result
        # The event content should NOT appear (only feel buckets)
        assert "event A content" not in result
    finally:
        await db.close()


@ASYNCIO
async def test_recall_memory_feel_channel_empty(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        method = _bind(MemoryToolsMixin.recall_memory, obj)
        result = await method(FakeEvent(), domain="feel")
        # Empty feel channel — just verify it returns a string and no crash.
        assert isinstance(result, str)
    finally:
        await db.close()


@ASYNCIO
async def test_recall_memory_importance_min_mode(tmp_path: Path):
    """importance_min >= 1 returns buckets with importance >= threshold."""
    db, obj = await _open_stack(tmp_path)
    try:
        await obj.writer.hold(SID, "low priority content", importance=2)
        await obj.writer.hold(SID, "medium priority content", importance=5)
        await obj.writer.hold(SID, "high priority content", importance=9)

        method = _bind(MemoryToolsMixin.recall_memory, obj)
        result = await method(FakeEvent(), importance_min=7)
        # Only the importance=9 bucket should appear
        assert "high priority content" in result
        assert "low priority content" not in result
        assert "medium priority content" not in result
    finally:
        await db.close()


@ASYNCIO
async def test_recall_memory_importance_min_no_match(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        await obj.writer.hold(SID, "common event content", importance=3)
        method = _bind(MemoryToolsMixin.recall_memory, obj)
        result = await method(FakeEvent(), importance_min=8)
        assert "common event content" not in result
        assert isinstance(result, str)
    finally:
        await db.close()


@ASYNCIO
async def test_recall_memory_no_query_no_special_mode(tmp_path: Path):
    """Empty query without special modes returns hint message."""
    db, obj = await _open_stack(tmp_path)
    try:
        method = _bind(MemoryToolsMixin.recall_memory, obj)
        result = await method(FakeEvent(), query="")
        # Returns string hint (non-empty)
        assert isinstance(result, str)
        assert len(result) > 0
    finally:
        await db.close()


# ===========================================================================
# reflect_memory
# ===========================================================================
@ASYNCIO
async def test_reflect_memory_returns_recent(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        await obj.writer.hold(SID, "memory one content")
        await obj.writer.hold(SID, "memory two content")
        method = _bind(MemoryToolsMixin.reflect_memory, obj)
        result = await method(FakeEvent())
        # Should include guidance text and at least one memory
        assert "forget_memory" in result
        assert "record_feel" in result
        assert "memory one content" in result or "memory two content" in result
    finally:
        await db.close()


@ASYNCIO
async def test_reflect_memory_empty_session(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        method = _bind(MemoryToolsMixin.reflect_memory, obj)
        result = await method(FakeEvent())
        # Empty session returns a string (guidance message)
        assert isinstance(result, str)
    finally:
        await db.close()


@ASYNCIO
async def test_reflect_memory_skips_resolved_and_pinned(tmp_path: Path):
    """Reflection only surfaces fresh dynamic non-resolved buckets."""
    db, obj = await _open_stack(tmp_path)
    try:
        h_resolved = await obj.writer.hold(SID, "resolved memory content")
        await obj.manager.update(SID, h_resolved.bucket_id, resolved=True)
        await obj.writer.hold(SID, "pinned core principle content", pinned=True)
        await obj.writer.hold(SID, "regular unresolved content")

        method = _bind(MemoryToolsMixin.reflect_memory, obj)
        result = await method(FakeEvent())
        # Resolved and pinned should not appear in the reflection list
        assert "resolved memory content" not in result
        assert "pinned core principle content" not in result
        assert "regular unresolved content" in result
    finally:
        await db.close()
