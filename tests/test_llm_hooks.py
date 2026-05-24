"""Tests for ``handlers.llm_hooks``.

We bind the hook methods to a minimal stand-in object so we don't need
a live AstrBot Context. Coverage focuses on:

- Memory injection populates ``req.system_prompt`` with header markers
- Hooks survive missing components (Property 7: never raise)
- Disabled-session config short-circuits injection
- Token-budget trim drops lowest-priority items
- Touch is called only on injected buckets
- Auto-record is skipped when the model already called record_memory
- Auto-record heuristic skips ACKs / commands / weather lookups
- Auto-record runs in a background task and doesn't block the hook
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from astrbot_plugin_ob_memory.core.embedding_service import EmbeddingService
from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.core.memory_writer import MemoryWriter
from astrbot_plugin_ob_memory.core.search_service import SearchService
from astrbot_plugin_ob_memory.core.surface_strategy import SurfaceStrategy
from astrbot_plugin_ob_memory.core.tagger import Tagger

from astrbot_plugin_ob_memory.handlers.llm_hooks import (
    MEMORY_BLOCK_FOOTER,
    MEMORY_BLOCK_HEADER,
    MemoryHooksMixin,
    _heuristic_should_auto_record,
    _injection_block,
    _trim_to_budget,
    _user_msg_from_request,
)
from astrbot_plugin_ob_memory.handlers.commands import format_digest_pairs

from astrbot_plugin_ob_memory.storage import Database, apply_migrations

ASYNCIO = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------
@dataclass
class FakeEvent:
    unified_msg_origin: str = "qq:GroupMessage:12345"
    message_str: str = ""


@dataclass
class FakeProviderRequest:
    """Minimal shape — matches the fields our hook touches."""

    prompt: str | None = None
    system_prompt: str = ""
    contexts: list[dict] = field(default_factory=list)


@dataclass
class FakeLLMResponse:
    role: str = "assistant"
    completion_text: str = ""
    tools_call_name: list[str] = field(default_factory=list)


class FakeAssembled(MemoryHooksMixin):
    """Lets us instantiate the mixin standalone for testing."""

    def __init__(self):
        self.manager: MemoryManager | None = None
        self.writer: MemoryWriter | None = None
        self.search: SearchService | None = None
        self.surface: SurfaceStrategy | None = None
        self.tagger: Tagger | None = None
        self.config: dict = {}


@dataclass
class StubLLMResp:
    completion_text: str = ""


class StubAnalyser:
    """Tagger stub with deterministic ``judge_worth_recording`` behaviour."""

    def __init__(self, should_record: bool = False, reason: str = "ok"):
        self.should_record = should_record
        self.reason = reason
        self.calls: list[tuple[str, str]] = []

    async def judge_worth_recording(
        self, user: str, assistant: str, *, session_id: str | None = None
    ) -> tuple[bool, str]:
        self.calls.append((user, assistant))
        return self.should_record, self.reason

    async def analyze(self, content: str, *, session_id: str | None = None) -> dict:
        return {
            "domain": ["未分类"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": [],
            "suggested_name": "",
            "importance": 5,
        }

    async def merge_content(self, old: str, new: str, *, session_id: str | None = None) -> str:
        return f"{old}\n{new}"


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
async def _open(tmp_path: Path) -> tuple[Database, FakeAssembled]:
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)

    obj = FakeAssembled()
    obj.manager = MemoryManager(db)
    obj.search = SearchService(obj.manager, embedding=None)
    obj.surface = SurfaceStrategy(obj.manager)
    obj.tagger = StubAnalyser(should_record=False)
    obj.writer = MemoryWriter(obj.manager, tagger=obj.tagger, embedding=None)
    return db, obj


# ===========================================================================
# Pure helpers
# ===========================================================================
def test_injection_block_empty():
    assert _injection_block([], []) == ""


def test_injection_block_contains_headers_and_items():
    from astrbot_plugin_ob_memory.core.models import new_bucket

    bucket = new_bucket("s", "core principle", name="一直要诚实")
    block = _injection_block([bucket], [])
    assert MEMORY_BLOCK_HEADER in block
    assert MEMORY_BLOCK_FOOTER in block
    assert "一直要诚实" in block
    assert bucket.id in block


def test_user_msg_from_request_prefers_prompt():
    req = FakeProviderRequest(
        prompt="hello world",
        contexts=[{"role": "user", "content": "ignored"}],
    )
    assert _user_msg_from_request(req) == "hello world"


def test_user_msg_from_request_falls_back_to_contexts():
    req = FakeProviderRequest(
        prompt=None,
        contexts=[
            {"role": "system", "content": "irrelevant"},
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "the latest user message"},
        ],
    )
    assert _user_msg_from_request(req) == "the latest user message"


def test_user_msg_from_request_handles_multipart():
    req = FakeProviderRequest(
        prompt=None,
        contexts=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see this image"},
                    {"type": "image_url", "image_url": "https://x"},
                ],
            }
        ],
    )
    assert "see this image" in _user_msg_from_request(req)


def test_heuristic_too_short():
    import re
    assert not _heuristic_should_auto_record(
        "hi", min_chars=30, skip_patterns=[re.compile(r"^/")]
    )


def test_heuristic_skip_pattern():
    import re
    pats = [re.compile(p) for p in (r"^/", r"天气")]
    assert not _heuristic_should_auto_record(
        "/forget last 3", min_chars=5, skip_patterns=pats
    )
    assert not _heuristic_should_auto_record(
        "今天上海的天气怎么样啊请告诉我", min_chars=5, skip_patterns=pats
    )


def test_heuristic_passes_substantive_message():
    import re
    pats = [re.compile(r"^/")]
    assert _heuristic_should_auto_record(
        "我今天面试到了一份梦寐以求的工作，非常激动",
        min_chars=20, skip_patterns=pats,
    )


@ASYNCIO
async def test_auto_record_skipped_when_model_used_record_diary(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        obj.tagger = StubAnalyser(should_record=True)
        obj.writer = MagicMock()
        obj.writer.hold_diary = MagicMock()
        response = FakeLLMResponse(
            completion_text="我帮你整理好了。",
            tools_call_name=["record_diary"],
        )

        await obj.memory_on_llm_response(
            FakeEvent(message_str="今天发生了很多事，我想写成日记。"),
            response,
        )
        await asyncio.sleep(0)

        obj.writer.hold_diary.assert_not_called()
    finally:
        await db.close()


@ASYNCIO
async def test_auto_record_uses_hold_diary_with_digest_text(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        obj.tagger = StubAnalyser(should_record=True, reason="important")
        obj.writer = MagicMock()
        obj.writer.hold_diary = MagicMock()
        obj.writer.hold_diary.return_value = MagicMock(entries=[], created=0, merged=0, failed=0)

        await obj._auto_record_task(
            "qq:GroupMessage:12345",
            "我今天拿到了 offer",
            "太好了，恭喜你。",
        )

        obj.writer.hold_diary.assert_called_once_with(
            "qq:GroupMessage:12345",
            format_digest_pairs([("我今天拿到了 offer", "太好了，恭喜你。")]),
        )
    finally:
        await db.close()


# ===========================================================================
# Memory injection
# ===========================================================================
@ASYNCIO
async def test_inject_pinned_bucket_appears_in_system_prompt(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        # Pin a "core principle" so it always surfaces.
        obj.config = {"max_search_results": 0, "max_surface_results": 5}
        await obj.manager.create_simple(
            "qq:GroupMessage:12345",
            "always be honest with her",
            name="诚实",
            pinned=True,
        )
        req = FakeProviderRequest(prompt="just chatting")
        await obj.memory_on_llm_request(FakeEvent(), req)
        assert MEMORY_BLOCK_HEADER in (req.system_prompt or "")
        assert "诚实" in req.system_prompt
    finally:
        await db.close()


@ASYNCIO
async def test_inject_keeps_existing_system_prompt(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        await obj.manager.create_simple(
            "qq:GroupMessage:12345", "core idea", pinned=True
        )
        req = FakeProviderRequest(
            prompt="x",
            system_prompt="You are a helpful assistant.",
        )
        await obj.memory_on_llm_request(FakeEvent(), req)
        assert "You are a helpful assistant." in req.system_prompt
        assert MEMORY_BLOCK_HEADER in req.system_prompt
    finally:
        await db.close()


@ASYNCIO
async def test_inject_no_op_when_session_has_no_memories(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        # Disable persona injection so this test isolates the memory
        # block path: with no buckets and no persona, system_prompt
        # must be untouched.
        obj.config = {"inject_memory_persona": False}
        req = FakeProviderRequest(
            prompt="x", system_prompt="original"
        )
        await obj.memory_on_llm_request(FakeEvent(), req)
        assert req.system_prompt == "original"
    finally:
        await db.close()


@ASYNCIO
async def test_inject_skips_disabled_session(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        await obj.manager.create_simple(
            "qq:GroupMessage:12345", "x", pinned=True
        )
        obj.config = {"disabled_sessions": ["qq:GroupMessage:12345"]}
        req = FakeProviderRequest(prompt="x")
        await obj.memory_on_llm_request(FakeEvent(), req)
        assert MEMORY_BLOCK_HEADER not in (req.system_prompt or "")
    finally:
        await db.close()


@ASYNCIO
async def test_inject_survives_search_failure(tmp_path: Path):
    """Property 7: a search exception must not raise out of the hook."""
    db, obj = await _open(tmp_path)
    try:
        await obj.manager.create_simple(
            "qq:GroupMessage:12345", "x", pinned=True
        )

        async def boom(*args, **kwargs):
            raise RuntimeError("search down")

        obj.search.search = boom  # type: ignore[assignment]
        req = FakeProviderRequest(prompt="anything")
        # Must not raise. Surface still runs and pinned bucket appears.
        await obj.memory_on_llm_request(FakeEvent(), req)
        assert MEMORY_BLOCK_HEADER in (req.system_prompt or "")
    finally:
        await db.close()


@ASYNCIO
async def test_inject_returns_silently_when_uninitialised():
    obj = FakeAssembled()  # nothing wired up
    req = FakeProviderRequest(prompt="x", system_prompt="orig")
    await obj.memory_on_llm_request(FakeEvent(), req)
    assert req.system_prompt == "orig"


@ASYNCIO
async def test_inject_touches_only_returned_buckets(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        # Create 3 pinned buckets but cap surface at 1 — only one is touched.
        for name in ("first", "second", "third"):
            await obj.manager.create_simple(
                "qq:GroupMessage:12345", name, name=name, pinned=True
            )

        obj.config = {"max_search_results": 0, "max_surface_results": 1}
        req = FakeProviderRequest(prompt="x")
        await obj.memory_on_llm_request(FakeEvent(), req)

        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        touched = [b for b in buckets if b.activation_count > 0]
        # Exactly 1 bucket got touched (the one in the surfaced output).
        assert len(touched) == 1
    finally:
        await db.close()


@ASYNCIO
async def test_memory_persona_injected_when_enabled_and_no_memories(tmp_path: Path):
    """Persona snippet appended to system_prompt even when no buckets match."""
    db, obj = await _open(tmp_path)
    try:
        # No buckets in DB → no memory block; persona should still inject.
        obj.config = {"inject_memory_persona": True}
        req = FakeProviderRequest(prompt="hi", system_prompt="you are X")
        await obj.memory_on_llm_request(FakeEvent(), req)

        # Persona marker present.
        assert "memory:" in (req.system_prompt or "")
        assert "remember_lightly" in (req.system_prompt or "")
        # User's original persona preserved at the top.
        assert req.system_prompt.startswith("you are X")
    finally:
        await db.close()


@ASYNCIO
async def test_memory_persona_disabled_when_toggle_off(tmp_path: Path):
    """``inject_memory_persona=False`` skips injection."""
    db, obj = await _open(tmp_path)
    try:
        obj.config = {"inject_memory_persona": False}
        req = FakeProviderRequest(prompt="hi", system_prompt="you are X")
        await obj.memory_on_llm_request(FakeEvent(), req)
        assert "remember_lightly" not in (req.system_prompt or "")
    finally:
        await db.close()


@ASYNCIO
async def test_memory_persona_custom_override(tmp_path: Path):
    """Custom ``memory_persona_text`` replaces the built-in default."""
    db, obj = await _open(tmp_path)
    try:
        custom = "CUSTOM_PERSONA_MARKER\nrules: be terse"
        obj.config = {
            "inject_memory_persona": True,
            "memory_persona_text": custom,
        }
        req = FakeProviderRequest(prompt="hi", system_prompt="")
        await obj.memory_on_llm_request(FakeEvent(), req)
        assert "CUSTOM_PERSONA_MARKER" in (req.system_prompt or "")
        # The built-in default's tokens must NOT leak through.
        assert "remember_lightly" not in (req.system_prompt or "")
    finally:
        await db.close()


@ASYNCIO
async def test_memory_persona_idempotent(tmp_path: Path):
    """Hook called twice doesn't duplicate the persona."""
    db, obj = await _open(tmp_path)
    try:
        obj.config = {"inject_memory_persona": True}
        req = FakeProviderRequest(prompt="hi", system_prompt="")
        await obj.memory_on_llm_request(FakeEvent(), req)
        once = req.system_prompt
        await obj.memory_on_llm_request(FakeEvent(), req)
        twice = req.system_prompt
        assert once == twice
        # Marker appears exactly once.
        assert (twice or "").count("remember_lightly") == 1
    finally:
        await db.close()


@ASYNCIO
async def test_inject_dedupes_surfaced_against_hits(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        # Pinned bucket whose content matches the user query.
        bucket = await obj.manager.create_simple(
            "qq:GroupMessage:12345",
            "she said she likes piano",
            name="piano",
            pinned=True,
        )
        obj.config = {"max_search_results": 5, "max_surface_results": 5}
        req = FakeProviderRequest(prompt="piano")
        await obj.memory_on_llm_request(FakeEvent(), req)
        # The bucket id must appear exactly once across the entire block.
        appearances = req.system_prompt.count(bucket.id)
        assert appearances == 1
    finally:
        await db.close()


# ===========================================================================
# trim_to_budget
# ===========================================================================
def test_trim_keeps_at_least_one_item():
    from astrbot_plugin_ob_memory.core.models import new_bucket

    a = new_bucket("s", "x" * 5000, name="big-1")
    b = new_bucket("s", "y" * 5000, name="big-2")
    surfaced, hits = _trim_to_budget([a, b], [], budget=1)
    # At least one survives even if both individually exceed the budget.
    assert len(surfaced) + len(hits) >= 1


def test_trim_drops_hits_first():
    from astrbot_plugin_ob_memory.core.models import new_bucket
    from astrbot_plugin_ob_memory.core.search_service import SearchHit

    s = new_bucket("s", "small", name="surfaced")
    big = new_bucket("s", "Z" * 4000, name="hit")
    h = SearchHit(bucket=big, score=10, via="keyword")
    surfaced, hits = _trim_to_budget([s], [h], budget=20)
    # Hits trimmed first; surfaced kept.
    assert len(surfaced) == 1
    assert len(hits) == 0


# ===========================================================================
# Auto-record (on_llm_response)
# ===========================================================================
@ASYNCIO
async def test_auto_record_skips_when_model_already_recorded(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        obj.tagger = StubAnalyser(should_record=True)
        obj.writer = MemoryWriter(obj.manager, tagger=obj.tagger, embedding=None)
        obj.config = {"auto_record_enabled": True}

        event = FakeEvent(message_str="long enough message about something significant")
        response = FakeLLMResponse(
            completion_text="acknowledged",
            tools_call_name=["record_memory"],
        )
        await obj.memory_on_llm_response(event, response)
        # No background task should have been scheduled — judge wasn't called.
        # Give event loop a tick to flush any (unwanted) created task.
        await asyncio.sleep(0.05)
        assert obj.tagger.calls == []
    finally:
        await db.close()


@ASYNCIO
async def test_auto_record_skipped_when_disabled(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        obj.tagger = StubAnalyser(should_record=True)
        obj.writer = MemoryWriter(obj.manager, tagger=obj.tagger, embedding=None)
        obj.config = {"auto_record_enabled": False}

        event = FakeEvent(message_str="a long substantive message about life things")
        response = FakeLLMResponse(completion_text="oh I see")
        await obj.memory_on_llm_response(event, response)
        await asyncio.sleep(0.05)
        assert obj.tagger.calls == []
    finally:
        await db.close()


@ASYNCIO
async def test_auto_record_skipped_for_short_user_msg(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        obj.tagger = StubAnalyser(should_record=True)
        obj.writer = MemoryWriter(obj.manager, tagger=obj.tagger, embedding=None)
        obj.config = {"auto_record_enabled": True, "auto_record_min_chars": 30}

        event = FakeEvent(message_str="hi")
        response = FakeLLMResponse(completion_text="hello")
        await obj.memory_on_llm_response(event, response)
        await asyncio.sleep(0.05)
        assert obj.tagger.calls == []
    finally:
        await db.close()


@ASYNCIO
async def test_auto_record_runs_when_judge_says_yes(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        obj.tagger = StubAnalyser(
            should_record=True, reason="user shared important fact"
        )
        obj.writer = MemoryWriter(obj.manager, tagger=obj.tagger, embedding=None)
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "per_turn",
            "auto_record_min_chars": 10,
        }

        event = FakeEvent(
            message_str="今天面试通过了那家公司的实习 offer，特别激动也有点紧张",
        )
        response = FakeLLMResponse(completion_text="恭喜！准备好了吗？")

        await obj.memory_on_llm_response(event, response)
        # The actual recording is in a background task — wait a beat.
        for _ in range(20):
            await asyncio.sleep(0.05)
            buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
            if buckets:
                break

        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        assert len(buckets) == 1
        assert "面试" in buckets[0].content
        assert obj.tagger.calls == [
            (
                "今天面试通过了那家公司的实习 offer，特别激动也有点紧张",
                "恭喜！准备好了吗？",
            )
        ]
    finally:
        await db.close()


@ASYNCIO
async def test_auto_record_skipped_when_judge_says_no(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        obj.tagger = StubAnalyser(should_record=False, reason="不重要")
        obj.writer = MemoryWriter(obj.manager, tagger=obj.tagger, embedding=None)
        obj.config = {"auto_record_enabled": True}

        event = FakeEvent(
            message_str="今天天气不错就随便聊聊大家最近吃什么了之类",
        )
        response = FakeLLMResponse(completion_text="哈哈是吗")

        await obj.memory_on_llm_response(event, response)
        await asyncio.sleep(0.2)
        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        assert buckets == []
    finally:
        await db.close()


@ASYNCIO
async def test_auto_record_does_not_block(tmp_path: Path):
    """The hook must return immediately even if judging is slow.

    We verify by giving the tagger a synthetic delay and asserting the
    hook itself returns much faster than that delay.
    """
    db, obj = await _open(tmp_path)
    try:
        slow = MagicMock()

        async def slow_judge(user, assistant, *, session_id=None):
            await asyncio.sleep(0.5)
            return False, "deliberately slow"

        slow.judge_worth_recording = slow_judge
        obj.tagger = slow
        obj.writer = MemoryWriter(obj.manager, tagger=StubAnalyser(), embedding=None)
        obj.config = {"auto_record_enabled": True}

        event = FakeEvent(
            message_str="a substantive message about life that triggers the heuristic"
        )
        response = FakeLLMResponse(completion_text="ack")

        import time

        t0 = time.monotonic()
        await obj.memory_on_llm_response(event, response)
        elapsed = time.monotonic() - t0
        # If we awaited the slow tagger the elapsed would be >= 0.5s.
        assert elapsed < 0.2
    finally:
        await db.close()


@ASYNCIO
async def test_auto_record_swallows_judge_exception(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        obj.tagger = MagicMock()

        async def boom(user, assistant, *, session_id=None):
            raise RuntimeError("tagger broken")

        obj.tagger.judge_worth_recording = boom
        obj.writer = MemoryWriter(obj.manager, tagger=StubAnalyser(), embedding=None)
        obj.config = {"auto_record_enabled": True}

        event = FakeEvent(
            message_str="long substantive message about something meaningful here",
        )
        response = FakeLLMResponse(completion_text="ok")
        await obj.memory_on_llm_response(event, response)
        # Wait long enough for the background task to attempt + fail.
        await asyncio.sleep(0.2)
        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        assert buckets == []  # no record landed; no crash either
    finally:
        await db.close()
