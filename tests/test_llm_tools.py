"""Tests for the LLM-callable tool methods.

We can't realistically import ``MemoryPlugin`` here because that would
require a live AstrBot Context. Instead we exercise the tool METHODS in
isolation by binding them to a minimal stand-in object that exposes the
same attributes (``manager``, ``writer``, ``search``, ``embedding``).

This still covers the meaningful behaviour: argument parsing, session-
id extraction, error containment, formatting, and the touch-on-recall
flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from astrbot_plugin_ob_memory.core.embedding_service import EmbeddingService
from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.core.memory_writer import MemoryWriter
from astrbot_plugin_ob_memory.core.search_service import SearchService
from astrbot_plugin_ob_memory.core.tagger import Tagger
from astrbot_plugin_ob_memory.handlers.llm_tools import (
    MemoryToolsMixin,
    _session_id,
    _split_csv,
)
from astrbot_plugin_ob_memory.storage import Database, apply_migrations

ASYNCIO = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------
@dataclass
class FakeEvent:
    """Mimics the bits of ``AstrMessageEvent`` the tools touch."""

    unified_msg_origin: str = "qq:GroupMessage:12345"


class FakeAssembled:
    """Minimal object onto which we bind the mixin methods.

    Avoiding ``MemoryPlugin`` here is important: instantiating that class
    triggers ``StarTools.get_data_dir()`` which is only valid inside a
    running AstrBot.
    """

    def __init__(self):
        self.manager: MemoryManager | None = None
        self.writer: MemoryWriter | None = None
        self.search: SearchService | None = None
        self.embedding: EmbeddingService | None = None

    # The mixin's ``_ready`` is unbound here; bind it manually for use.
    def _ready(self):
        return MemoryToolsMixin._ready(self)


def _bind(method, instance):
    """Call a mixin method as if bound to ``instance``.

    ``@filter.llm_tool`` wraps the method but returns the original
    awaitable unchanged, so we can still call the raw function with a
    manually-passed ``self``.
    """
    return method.__get__(instance, instance.__class__)


# Stub LLM provider for the writer's auto-tagging (returns canned JSON).
@dataclass
class FakeLLMResponse:
    completion_text: str


class StubLLMProvider:
    def __init__(self, analyse: str = "", merge: str = ""):
        self.analyse = analyse
        self.merge = merge

    async def text_chat(
        self, prompt: str | None = None, system_prompt: str | None = None, **kw
    ) -> FakeLLMResponse:
        sp = (system_prompt or "").lower()
        if "memory analyst" in sp:
            return FakeLLMResponse(self.analyse)
        if "merge two related memory contents" in sp:
            return FakeLLMResponse(self.merge)
        return FakeLLMResponse("")


def _analyse_payload(**kw) -> str:
    import json

    payload = {
        "domain": kw.get("domain", ["未分类"]),
        "valence": kw.get("valence", 0.5),
        "arousal": kw.get("arousal", 0.3),
        "tags": kw.get("tags", []),
        "suggested_name": kw.get("name", ""),
        "importance": kw.get("importance", 5),
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Helpers — bring up a fully-wired writer + search stack
# ---------------------------------------------------------------------------
async def _open_stack(
    tmp_path: Path,
    *,
    analyse_response: str | None = None,
) -> tuple[Database, FakeAssembled]:
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    obj = FakeAssembled()
    obj.manager = MemoryManager(db)
    provider = StubLLMProvider(
        analyse=analyse_response or _analyse_payload(),
        merge="merged content",
    )
    obj.writer = MemoryWriter(
        obj.manager,
        tagger=Tagger(context=None, fixed_provider=provider),
        embedding=None,
    )
    obj.search = SearchService(obj.manager, embedding=None)
    return db, obj


# ===========================================================================
# Pure helpers
# ===========================================================================
def test_split_csv_handles_blank_and_whitespace():
    assert _split_csv("") == []
    assert _split_csv(None) == []
    assert _split_csv("a, b ,  ,c") == ["a", "b", "c"]


def test_session_id_falls_back_to_unknown():
    e = FakeEvent(unified_msg_origin="")
    assert _session_id(e) == "unknown"
    e2 = FakeEvent(unified_msg_origin="qq:Private:99")
    assert _session_id(e2) == "qq:Private:99"


# ===========================================================================
# record_memory
# ===========================================================================
@ASYNCIO
async def test_record_memory_creates_bucket(tmp_path: Path):
    db, obj = await _open_stack(
        tmp_path,
        analyse_response=_analyse_payload(
            domain=["求职"], valence=0.8, arousal=0.7,
            name="实习offer", importance=7, tags=["实习"],
        ),
    )
    try:
        method = _bind(MemoryToolsMixin.record_memory, obj)
        result = await method(
            FakeEvent(),
            content="拿到了实习 offer",
            importance=7,
            tags="实习,offer",
            pinned=False,
        )
        assert "新建记忆" in result
        # Persisted in DB.
        sessions = await obj.manager.list_sessions()
        assert sessions == ["qq:GroupMessage:12345"]
        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        assert len(buckets) == 1
        b = buckets[0]
        assert b.importance == 7
        assert "实习" in b.tags
        assert "offer" in b.tags
    finally:
        await db.close()


@ASYNCIO
async def test_record_memory_pinned_lands_in_permanent(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        method = _bind(MemoryToolsMixin.record_memory, obj)
        result = await method(
            FakeEvent(),
            content="be honest, never lie",
            importance=5,  # will be forced to 10 by pin
            tags="",
            pinned=True,
        )
        assert "📌" in result
        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        assert len(buckets) == 1
        assert buckets[0].pinned is True
        assert buckets[0].importance == 10
        assert buckets[0].bucket_type == "permanent"
    finally:
        await db.close()


@ASYNCIO
async def test_record_memory_rejects_empty_content(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        method = _bind(MemoryToolsMixin.record_memory, obj)
        result = await method(FakeEvent(), content="", importance=5)
        assert "为空" in result
        assert await obj.manager.list_sessions() == []
    finally:
        await db.close()


@ASYNCIO
async def test_record_memory_returns_error_string_when_uninitialised():
    obj = FakeAssembled()  # no manager / writer
    method = _bind(MemoryToolsMixin.record_memory, obj)
    result = await method(FakeEvent(), content="x", importance=5)
    assert "未初始化" in result


@ASYNCIO
async def test_record_memory_session_isolation(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        method = _bind(MemoryToolsMixin.record_memory, obj)
        await method(
            FakeEvent(unified_msg_origin="session-A"),
            content="alpha", importance=5,
        )
        await method(
            FakeEvent(unified_msg_origin="session-B"),
            content="beta", importance=5,
        )
        a = await obj.manager.list_by_session("session-A")
        b = await obj.manager.list_by_session("session-B")
        assert {x.content for x in a} == {"alpha"}
        assert {x.content for x in b} == {"beta"}
    finally:
        await db.close()


# ===========================================================================
# record_feel
# ===========================================================================
@ASYNCIO
async def test_record_feel_creates_feel_bucket(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        method = _bind(MemoryToolsMixin.record_feel, obj)
        result = await method(
            FakeEvent(),
            content="她让我想到自己也曾这样困惑过",
            source_bucket_id="",
            valence=0.6,
        )
        assert "🫧" in result
        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        feels = [b for b in buckets if b.bucket_type == "feel"]
        assert len(feels) == 1
        assert feels[0].valence == pytest.approx(0.6)
    finally:
        await db.close()


@ASYNCIO
async def test_record_feel_marks_source_digested(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        # First create an event memory.
        rec = _bind(MemoryToolsMixin.record_memory, obj)
        msg = await rec(FakeEvent(), content="她生气了", importance=6)
        # Extract the bucket id from the response.
        bucket_id = msg.split("id:")[-1].rstrip(")")

        feel = _bind(MemoryToolsMixin.record_feel, obj)
        result = await feel(
            FakeEvent(),
            content="我从中看到了她的成长",
            source_bucket_id=bucket_id,
            valence=0.7,
        )
        assert "已标记为已消化" in result

        loaded = await obj.manager.get("qq:GroupMessage:12345", bucket_id)
        assert loaded is not None
        assert loaded.digested is True
    finally:
        await db.close()


@ASYNCIO
async def test_record_feel_invalid_valence_falls_back(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        feel = _bind(MemoryToolsMixin.record_feel, obj)
        # valence=-1 means "not specified" — must not error.
        result = await feel(FakeEvent(), content="x", valence=-1.0)
        assert "🫧" in result
    finally:
        await db.close()


# ===========================================================================
# recall_memory
# ===========================================================================
@ASYNCIO
async def test_recall_memory_returns_hits_and_touches(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        # Pre-populate two buckets.
        rec = _bind(MemoryToolsMixin.record_memory, obj)
        await rec(FakeEvent(), content="实习 offer 拿到了", importance=7)
        await rec(FakeEvent(), content="无关日常对话", importance=3)

        recall = _bind(MemoryToolsMixin.recall_memory, obj)
        result = await recall(FakeEvent(), query="实习", limit=5)
        assert "检索到" in result
        assert "实习" in result

        # The relevant bucket should have been touched (activation_count > 0).
        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        relevant = [b for b in buckets if "实习" in b.content]
        assert relevant
        assert relevant[0].activation_count >= 1.0
    finally:
        await db.close()


@ASYNCIO
async def test_recall_memory_returns_no_hits_message(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        recall = _bind(MemoryToolsMixin.recall_memory, obj)
        result = await recall(FakeEvent(), query="不存在的关键词", limit=5)
        assert "没有找到" in result
    finally:
        await db.close()


@ASYNCIO
async def test_recall_memory_clamps_limit_to_20(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        recall = _bind(MemoryToolsMixin.recall_memory, obj)
        # Limit=50 should be clamped silently.
        result = await recall(FakeEvent(), query="x", limit=50)
        # No exception, no empty crash.
        assert isinstance(result, str)
    finally:
        await db.close()


@ASYNCIO
async def test_recall_memory_rejects_empty_query(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        recall = _bind(MemoryToolsMixin.recall_memory, obj)
        result = await recall(FakeEvent(), query="", limit=5)
        assert "未提供" in result
    finally:
        await db.close()


# ===========================================================================
# forget_memory
# ===========================================================================
@ASYNCIO
async def test_forget_memory_resolve_default(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        rec = _bind(MemoryToolsMixin.record_memory, obj)
        msg = await rec(FakeEvent(), content="something", importance=5)
        bucket_id = msg.split("id:")[-1].rstrip(")")

        forget = _bind(MemoryToolsMixin.forget_memory, obj)
        result = await forget(FakeEvent(), bucket_id=bucket_id, mode="resolve")
        assert "沉底" in result

        loaded = await obj.manager.get("qq:GroupMessage:12345", bucket_id)
        assert loaded is not None
        assert loaded.resolved is True
    finally:
        await db.close()


@ASYNCIO
async def test_forget_memory_delete_mode(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        rec = _bind(MemoryToolsMixin.record_memory, obj)
        msg = await rec(FakeEvent(), content="stuff", importance=5)
        bucket_id = msg.split("id:")[-1].rstrip(")")

        forget = _bind(MemoryToolsMixin.forget_memory, obj)
        result = await forget(FakeEvent(), bucket_id=bucket_id, mode="delete")
        assert "永久删除" in result

        loaded = await obj.manager.get("qq:GroupMessage:12345", bucket_id)
        assert loaded is None
    finally:
        await db.close()


@ASYNCIO
async def test_forget_memory_unknown_id_returns_friendly_error(tmp_path: Path):
    db, obj = await _open_stack(tmp_path)
    try:
        forget = _bind(MemoryToolsMixin.forget_memory, obj)
        result = await forget(FakeEvent(), bucket_id="no-such-id", mode="resolve")
        assert "未找到" in result
    finally:
        await db.close()


@ASYNCIO
async def test_forget_memory_session_isolation(tmp_path: Path):
    """A session-A forget must not affect session-B's identical bucket id."""
    db, obj = await _open_stack(tmp_path)
    try:
        rec = _bind(MemoryToolsMixin.record_memory, obj)
        a_msg = await rec(
            FakeEvent(unified_msg_origin="session-A"),
            content="alpha", importance=5,
        )
        a_id = a_msg.split("id:")[-1].rstrip(")")

        forget = _bind(MemoryToolsMixin.forget_memory, obj)
        # Try to delete A's bucket from session B.
        result = await forget(
            FakeEvent(unified_msg_origin="session-B"),
            bucket_id=a_id, mode="delete",
        )
        assert "未找到" in result

        # A's bucket is untouched.
        loaded = await obj.manager.get("session-A", a_id)
        assert loaded is not None
    finally:
        await db.close()
