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
    _get_recent_pairs_buffer,
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
class FakeGroupEvent:
    """Group-chat event stub with a known sender name + ``is_private_chat``
    returning ``False`` so the speaker-enrichment path in
    ``_maybe_schedule_auto_record`` lights up.

    Models the real AstrBot ``AstrMessageEvent`` surface that the
    handler relies on (``get_sender_name`` / ``is_private_chat`` /
    ``message_str``) without dragging the framework in.
    """

    unified_msg_origin: str = "qq:GroupMessage:12345"
    message_str: str = ""
    sender_name: str = "小明"
    sender_id: str = "1001"

    def get_sender_name(self) -> str:
        return self.sender_name

    def get_sender_id(self) -> str:
        return self.sender_id

    def is_private_chat(self) -> bool:
        return False


@dataclass
class FakePrivateEvent:
    """Private-chat counterpart used to prove the enrichment path is a
    no-op for 1-to-1 conversations (no risk of changing the wire
    format for existing private-chat users)."""

    unified_msg_origin: str = "qq:FriendMessage:7890"
    message_str: str = ""
    sender_name: str = "alice"
    sender_id: str = "7890"

    def get_sender_name(self) -> str:
        return self.sender_name

    def get_sender_id(self) -> str:
        return self.sender_id

    def is_private_chat(self) -> bool:
        return True


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
            group_context=False,
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
async def test_every_n_turns_counter_increments_until_threshold(tmp_path: Path):
    """First N-1 turns just bump the counter, no summary fires."""
    db, obj = await _open(tmp_path)
    try:
        obj.writer = MagicMock()
        obj.writer.hold_diary = MagicMock()
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "every_n_turns",
            "auto_record_every_n_turns": 3,
        }

        # First 2 turns: counter goes 1, 2 — no summary call yet.
        for _ in range(2):
            event = FakeEvent(message_str="something to say")
            response = FakeLLMResponse(completion_text="reply text")
            await obj.memory_on_llm_response(event, response)
            await asyncio.sleep(0.05)

        obj.writer.hold_diary.assert_not_called()
        assert (
            await obj.manager.get_auto_record_counter("qq:GroupMessage:12345")
            == 2
        )
    finally:
        await db.close()


@ASYNCIO
async def test_every_n_turns_triggers_summary_at_threshold(tmp_path: Path):
    """N-th turn must invoke hold_diary with the last N user/assistant pairs.

    Regression for the ``_get_conversation_history`` tuple-unpacking bug:
    the helper returns ``(history, debug_info)`` and the auto-summary path
    must unpack both, otherwise ``_extract_pairs`` iterates over a 2-tuple
    and produces no pairs, silently dropping the summary.
    """
    db, obj = await _open(tmp_path)
    try:
        recorded_args: list[tuple[str, str]] = []

        async def fake_hold_diary(session_id, text, **kwargs):
            recorded_args.append((session_id, text))
            return MagicMock(entries=[], created=1, merged=0, failed=0)

        obj.writer = MagicMock()
        obj.writer.hold_diary = fake_hold_diary
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "every_n_turns",
            "auto_record_every_n_turns": 2,
        }

        # Stub _get_conversation_history on the instance — it normally lives
        # on CommandsMixin which the test FakeAssembled doesn't include.
        async def fake_history(event):
            return (
                [
                    {"role": "user", "content": "今天面试拿到 offer"},
                    {"role": "assistant", "content": "太棒了，恭喜你"},
                    {"role": "user", "content": "下周一就入职"},
                    {"role": "assistant", "content": "记住啦"},
                ],
                "test history",
            )

        obj._get_conversation_history = fake_history  # type: ignore[attr-defined]

        # Turn 1: counter goes to 1, no summary.
        await obj.memory_on_llm_response(
            FakeEvent(message_str="今天面试拿到 offer"),
            FakeLLMResponse(completion_text="太棒了，恭喜你"),
        )
        await asyncio.sleep(0.05)
        assert recorded_args == []

        # Turn 2: counter hits 2 = threshold, summary fires.
        await obj.memory_on_llm_response(
            FakeEvent(message_str="下周一就入职"),
            FakeLLMResponse(completion_text="记住啦"),
        )
        # background task — wait for it.
        for _ in range(20):
            if recorded_args:
                break
            await asyncio.sleep(0.05)

        assert len(recorded_args) == 1
        session_id, text = recorded_args[0]
        assert session_id == "qq:GroupMessage:12345"
        # The summary text should contain content from both pairs.
        assert "offer" in text
        assert "入职" in text

        # Counter should be reset to 0 after firing.
        assert (
            await obj.manager.get_auto_record_counter("qq:GroupMessage:12345")
            == 0
        )
    finally:
        await db.close()


@ASYNCIO
async def test_every_n_turns_skips_turn_when_model_used_tool(tmp_path: Path):
    """Model-invoked record_memory must NOT count toward the threshold (so we
    don't re-summarise content the model already captured), but the previously
    accumulated progress is preserved — earlier versions reset the counter to
    0, which made N=10/15 effectively unreachable in chatty sessions where the
    model autonomously records once every few turns.
    """
    db, obj = await _open(tmp_path)
    try:
        obj.writer = MagicMock()
        obj.writer.hold_diary = MagicMock()
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "every_n_turns",
            "auto_record_every_n_turns": 5,
        }

        # Bump the counter twice.
        for _ in range(2):
            await obj.memory_on_llm_response(
                FakeEvent(message_str="something"),
                FakeLLMResponse(completion_text="ok"),
            )
            await asyncio.sleep(0.01)

        assert (
            await obj.manager.get_auto_record_counter("qq:GroupMessage:12345")
            == 2
        )

        # Now a turn where the model called record_memory — counter must
        # stay at 2 (this turn isn't counted, but earlier progress is kept).
        await obj.memory_on_llm_response(
            FakeEvent(message_str="something memorable"),
            FakeLLMResponse(
                completion_text="recorded",
                tools_call_name=["record_memory"],
            ),
        )
        await asyncio.sleep(0.01)
        assert (
            await obj.manager.get_auto_record_counter("qq:GroupMessage:12345")
            == 2
        )

        # writer.hold_diary must NOT have been called via the skip path.
        obj.writer.hold_diary.assert_not_called()
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


@ASYNCIO
async def test_every_n_turns_falls_back_to_rolling_buffer_on_empty_history(
    tmp_path: Path,
):
    """Group-chat-friendly fallback: when ``_get_conversation_history``
    returns empty (e.g. adapter doesn't track group cids), the auto-summary
    must still fire using the in-memory ``(user, assistant)`` buffer that
    ``_maybe_schedule_auto_record`` populates each turn.

    Regression for "群聊那边好像又没有按照计数器自动总结了" — when
    ``conversation_manager`` had no history for the group, the previous
    implementation silently dropped the summary at DEBUG level.
    """
    db, obj = await _open(tmp_path)
    try:
        recorded_args: list[tuple[str, str]] = []

        async def fake_hold_diary(session_id, text, **kwargs):
            recorded_args.append((session_id, text))
            return MagicMock(entries=[], created=1, merged=0, failed=0)

        obj.writer = MagicMock()
        obj.writer.hold_diary = fake_hold_diary
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "every_n_turns",
            "auto_record_every_n_turns": 2,
        }

        # Simulate the group-chat case: history fetch returns empty.
        async def empty_history(event):
            return ([], "no cid for group adapter")

        obj._get_conversation_history = empty_history  # type: ignore[attr-defined]

        # Turn 1: bump counter to 1, fills buffer with the first pair.
        await obj.memory_on_llm_response(
            FakeEvent(message_str="群里有人在吗"),
            FakeLLMResponse(completion_text="我在的"),
        )
        await asyncio.sleep(0.05)
        assert recorded_args == []  # not yet at threshold

        # Turn 2: hits threshold = 2, summary must fire from the buffer.
        await obj.memory_on_llm_response(
            FakeEvent(message_str="我今天去看了那个展览"),
            FakeLLMResponse(completion_text="听起来很有意思"),
        )
        for _ in range(20):
            if recorded_args:
                break
            await asyncio.sleep(0.05)

        assert len(recorded_args) == 1, (
            "auto-summary should have fired from the buffer fallback"
        )
        session_id, text = recorded_args[0]
        assert session_id == "qq:GroupMessage:12345"
        # Buffer must include content from both turns.
        assert "群里" in text or "看了那个展览" in text
        assert "展览" in text or "听起来很有意思" in text
    finally:
        await db.close()


# ===========================================================================
# Group-chat speaker enrichment — "不认人" regression suite.
#
# In a multi-speaker group chat the digest LLM previously received
# every user message tagged uniformly as ``对方(用户)`` (see
# ``format_digest_pairs``). With no per-message speaker attribution,
# the LLM cross-attributed facts (e.g. "Alice said X" got merged into
# "Bob said X"), which surfaced as the user-reported "群聊里 bot 不认人,
# 乱记" symptom.
#
# The fix decorates user messages with ``[sender_name] `` before they
# hit the rolling buffer (group chats only), so downstream consumers
# (auto-summary, per-turn auto-record) keep their existing
# ``(user_msg, assistant_msg)`` shape but each user_msg now carries
# its speaker. These tests pin that contract end-to-end.
# ===========================================================================
@ASYNCIO
async def test_buffer_decorates_user_msg_with_speaker_in_group_chat(
    tmp_path: Path,
):
    """Rolling buffer must store ``[小明] ...`` for group-chat turns so
    the digest LLM can distinguish speakers later when it's fed multiple
    turns at once.

    Without this, two distinct people in a group both end up as
    ``对方(用户)说: ...`` and the LLM has no way to attribute anything.
    """
    db, obj = await _open(tmp_path)
    try:
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "every_n_turns",
            "auto_record_every_n_turns": 99,  # never fire summary in this test
        }
        # Empty conversation_manager history so the buffer is the
        # only place we can observe the enrichment.
        async def empty_history(event):
            return ([], "no cid for group adapter")

        obj._get_conversation_history = empty_history  # type: ignore[attr-defined]

        await obj.memory_on_llm_response(
            FakeGroupEvent(message_str="我今天拿到 offer 了", sender_name="小明"),
            FakeLLMResponse(completion_text="恭喜！"),
        )
        await asyncio.sleep(0.05)
        await obj.memory_on_llm_response(
            FakeGroupEvent(message_str="我家狗子刚没了", sender_name="小红"),
            FakeLLMResponse(completion_text="节哀…"),
        )
        await asyncio.sleep(0.05)

        # The buffer is keyed by the resolved memory session_id. For
        # this test the default scope_mode produces conv:None or umo-
        # based id, so just look at the only populated bucket.
        buffers = [
            buf for buf in getattr(obj, "_recent_pairs", {}).values()
            if buf
        ]
        assert len(buffers) == 1, (
            "expected exactly one populated buffer for the group session"
        )
        pairs = list(buffers[0])
        # Both user messages must carry their speaker tag.
        assert pairs[0][0] == "[小明] 我今天拿到 offer 了"
        assert pairs[1][0] == "[小红] 我家狗子刚没了"
        # Assistant replies are untouched (only one AI; ``format_digest_pairs``
        # already labels them as ``我(AI)``).
        assert pairs[0][1] == "恭喜！"
        assert pairs[1][1] == "节哀…"
    finally:
        await db.close()


@ASYNCIO
async def test_buffer_leaves_private_chat_user_msg_alone(tmp_path: Path):
    """Private chats must keep the wire format bit-for-bit unchanged
    so existing summaries / configurations don't suddenly start
    receiving ``[alice] ...`` prefixes that they never asked for.

    Only the group-chat path enriches.
    """
    db, obj = await _open(tmp_path)
    try:
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "every_n_turns",
            "auto_record_every_n_turns": 99,
        }
        async def empty_history(event):
            return ([], "no cid")

        obj._get_conversation_history = empty_history  # type: ignore[attr-defined]

        await obj.memory_on_llm_response(
            FakePrivateEvent(message_str="我今天好累"),
            FakeLLMResponse(completion_text="休息一下吧"),
        )
        await asyncio.sleep(0.05)

        buffers = [
            buf for buf in getattr(obj, "_recent_pairs", {}).values()
            if buf
        ]
        assert len(buffers) == 1
        pairs = list(buffers[0])
        # No ``[alice]`` prefix — private chat content must be untouched.
        assert pairs[0] == ("我今天好累", "休息一下吧")
    finally:
        await db.close()


@ASYNCIO
async def test_group_summary_propagates_speaker_tags_into_digest_text(
    tmp_path: Path,
):
    """End-to-end: when auto-summary fires in a group chat, the text
    handed to ``writer.hold_diary`` (which is what the digest LLM sees)
    must contain the speaker tags, otherwise the whole enrichment
    pipeline is decorative noise.
    """
    db, obj = await _open(tmp_path)
    try:
        recorded_args: list[tuple[str, str]] = []

        async def fake_hold_diary(session_id, text, **kwargs):
            recorded_args.append((session_id, text))
            return MagicMock(entries=[], created=1, merged=0, failed=0)

        obj.writer = MagicMock()
        obj.writer.hold_diary = fake_hold_diary
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "every_n_turns",
            "auto_record_every_n_turns": 2,
        }
        async def empty_history(event):
            return ([], "no cid for group adapter")

        obj._get_conversation_history = empty_history  # type: ignore[attr-defined]

        await obj.memory_on_llm_response(
            FakeGroupEvent(message_str="我今天拿到 offer 了", sender_name="小明"),
            FakeLLMResponse(completion_text="恭喜！"),
        )
        await asyncio.sleep(0.05)
        await obj.memory_on_llm_response(
            FakeGroupEvent(message_str="我家狗子刚没了", sender_name="小红"),
            FakeLLMResponse(completion_text="节哀…"),
        )
        for _ in range(20):
            if recorded_args:
                break
            await asyncio.sleep(0.05)

        assert len(recorded_args) == 1, "auto-summary should have fired"
        _, digest_text = recorded_args[0]
        # The digest LLM input must surface both speakers in an
        # unambiguous way: the speaker tag is lifted from the in-body
        # ``[小明] ...`` form into the framing prefix
        # ``对方(小明)说: ...`` so it cannot be misparsed as an
        # in-message address ("称呼") by the LLM.
        assert "对方(小明)说: 我今天拿到 offer 了" in digest_text
        assert "对方(小红)说: 我家狗子刚没了" in digest_text
        # Sanity: the raw [name] body form must not leak through.
        assert "[小明]" not in digest_text
        assert "[小红]" not in digest_text
    finally:
        await db.close()


# ===========================================================================
# Group-chat perspective consistency
#
# Ensures the group_context flag is threaded to hold_diary so the
# digest LLM gets the GROUP_DIGEST_ADDENDUM instructing it to use
# third-person speaker names rather than ambiguous "你".
# ===========================================================================
@ASYNCIO
async def test_auto_summary_passes_group_context_true_for_group_events(tmp_path: Path):
    """_auto_summary_task must pass group_context=True when event is a
    group chat, so hold_diary appends GROUP_DIGEST_ADDENDUM to the
    system prompt."""
    db, obj = await _open(tmp_path)
    try:
        captured_kwargs: list[dict] = []

        async def spy_hold_diary(session_id, text, **kwargs):
            captured_kwargs.append(kwargs)
            return MagicMock(entries=[], created=1, merged=0, failed=0)

        obj.writer = MagicMock()
        obj.writer.hold_diary = spy_hold_diary
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "every_n_turns",
            "auto_record_every_n_turns": 2,
        }

        async def empty_history(event):
            return ([], "no cid for group adapter")

        obj._get_conversation_history = empty_history  # type: ignore[attr-defined]

        group_event = FakeGroupEvent(message_str="我拿到 offer 了")
        response = MagicMock(completion_text="恭喜你")

        # Turn 1
        await obj.memory_on_llm_response(group_event, response)
        # Turn 2 → triggers summary
        group_event2 = FakeGroupEvent(
            message_str="下周入职", sender_name="小红", sender_id="2002"
        )
        response2 = MagicMock(completion_text="期待")
        await obj.memory_on_llm_response(group_event2, response2)

        # Give asyncio tasks a chance to run
        import asyncio
        await asyncio.sleep(0.1)

        assert len(captured_kwargs) >= 1
        assert captured_kwargs[0].get("group_context") is True
    finally:
        await db.close()


@ASYNCIO
async def test_auto_summary_passes_group_context_false_for_private_events(tmp_path: Path):
    """Private-chat events must pass group_context=False so the
    familiar 你/我 perspective is preserved."""
    db, obj = await _open(tmp_path)
    try:
        captured_kwargs: list[dict] = []

        async def spy_hold_diary(session_id, text, **kwargs):
            captured_kwargs.append(kwargs)
            return MagicMock(entries=[], created=1, merged=0, failed=0)

        obj.writer = MagicMock()
        obj.writer.hold_diary = spy_hold_diary
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "every_n_turns",
            "auto_record_every_n_turns": 2,
        }

        async def fake_history(event):
            return ([
                {"role": "user", "content": "今天好累"},
                {"role": "assistant", "content": "休息一下"},
                {"role": "user", "content": "嗯"},
                {"role": "assistant", "content": "晚安"},
            ], "ok")

        obj._get_conversation_history = fake_history  # type: ignore[attr-defined]

        private_event = FakePrivateEvent(message_str="今天好累")
        response = MagicMock(completion_text="休息一下")
        await obj.memory_on_llm_response(private_event, response)

        private_event2 = FakePrivateEvent(message_str="嗯")
        response2 = MagicMock(completion_text="晚安")
        await obj.memory_on_llm_response(private_event2, response2)

        import asyncio
        await asyncio.sleep(0.1)

        assert len(captured_kwargs) >= 1
        assert captured_kwargs[0].get("group_context") is False
    finally:
        await db.close()


@ASYNCIO
async def test_per_turn_auto_record_passes_group_context_for_group(tmp_path: Path):
    """Per-turn mode: _auto_record_task receives group_context from the
    event's is_private_chat status."""
    db, obj = await _open(tmp_path)
    try:
        obj.tagger = StubAnalyser(should_record=True, reason="important")
        obj.writer = MagicMock()
        obj.writer.hold_diary = MagicMock()
        obj.writer.hold_diary.return_value = MagicMock(entries=[], created=0, merged=0, failed=0)

        await obj._auto_record_task(
            "qq:GroupMessage:12345",
            "[小明] 我今天拿到了 offer",
            "太好了，恭喜你。",
            group_context=True,
        )

        obj.writer.hold_diary.assert_called_once_with(
            "qq:GroupMessage:12345",
            format_digest_pairs([("[小明] 我今天拿到了 offer", "太好了，恭喜你。")]),
            group_context=True,
        )
    finally:
        await db.close()
