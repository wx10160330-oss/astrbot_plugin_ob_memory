"""Tests for ``handlers.commands`` — the ``/memory`` subcommand handlers.

We bind the subcommand methods to a stand-in object (FakeAssembled) so we
don't depend on a live AstrBot Context. Each subcommand is an async
generator yielding ``MessageEventResult``-shaped objects; we collect
them by iterating the generator manually.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrbot_plugin_ob_memory.handlers.commands import (
    DESTRUCTIVE_TTL_SECONDS,
    MemoryCommandsMixin,
    _clean_import_text,
    _convert_platform_history,
    _event_is_group_chat,
    _extract_pairs_from_astrbot_jsonl,
    _is_admin,
    _pending,
    decorate_user_msg_with_speaker,
    format_digest_pairs,
)
from astrbot_plugin_ob_memory.core.decay_engine import DecayConfig, DecayEngine
from astrbot_plugin_ob_memory.core.embedding_service import EmbeddingService
from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.core.memory_writer import MemoryWriter
from astrbot_plugin_ob_memory.core.search_service import SearchService
from astrbot_plugin_ob_memory.core.tagger import Tagger
from astrbot_plugin_ob_memory.storage import Database, apply_migrations

ASYNCIO = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------
@dataclass
class FakeResult:
    text: str

    def __repr__(self):
        return f"FakeResult({self.text!r})"


@dataclass
class FakeEvent:
    unified_msg_origin: str = "qq:GroupMessage:12345"
    is_admin: bool = False

    def plain_result(self, text: str) -> FakeResult:
        return FakeResult(text=text)


class FakeAssembled(MemoryCommandsMixin):
    """Lets us exercise the mixin without a real plugin."""

    def __init__(self):
        self.manager: MemoryManager | None = None
        self.search: SearchService | None = None
        self.surface = None
        self.decay: DecayEngine | None = None
        self.embedding: EmbeddingService | None = None
        self.writer: MemoryWriter | None = None
        self.context = SimpleNamespace(conversation_manager=None)


async def _open(tmp_path: Path) -> tuple[Database, FakeAssembled]:
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)

    obj = FakeAssembled()
    obj.manager = MemoryManager(db)
    obj.search = SearchService(obj.manager, embedding=None)
    obj.embedding = None
    obj.writer = MemoryWriter(obj.manager, tagger=None, embedding=None)
    obj.decay = DecayEngine(obj.manager, DecayConfig(check_interval_hours=0))
    return db, obj


async def _collect(gen) -> list[FakeResult]:
    """Drain an async generator into a list."""
    out: list[FakeResult] = []
    async for r in gen:
        out.append(r)
    return out


# ===========================================================================
# Helpers
# ===========================================================================
def test_is_admin_via_attribute():
    e = FakeEvent(is_admin=True)
    assert _is_admin(e) is True


def test_is_admin_default_false():
    e = FakeEvent(is_admin=False)
    assert _is_admin(e) is False


def test_pending_register_expires():
    import time
    from astrbot_plugin_ob_memory.handlers.commands import _Pending

    p = _Pending()
    p.set("s", "delete", "abc")
    # Force-expire
    p._items[("s", "delete")] = ("abc", time.time() - DESTRUCTIVE_TTL_SECONDS - 1)
    assert p.pop_if_match("s", "delete", "abc") is False


def test_pending_match_consumes():
    from astrbot_plugin_ob_memory.handlers.commands import _Pending

    p = _Pending()
    p.set("s", "delete", "abc")
    assert p.pop_if_match("s", "delete", "abc") is True
    # Second confirm fails because the pending was consumed.
    assert p.pop_if_match("s", "delete", "abc") is False


def test_pending_target_must_match():
    from astrbot_plugin_ob_memory.handlers.commands import _Pending

    p = _Pending()
    p.set("s", "delete", "abc")
    assert p.pop_if_match("s", "delete", "xyz") is False


def test_clean_import_text_removes_rag_and_system_noise():
    raw = "[发送时间: 2026-05-16 14:38]\n你好\n<RAG-Faiss-Memory>old memory</RAG-Faiss-Memory>"
    assert _clean_import_text(raw) == "你好"
    assert _clean_import_text("当前时间是: 2026年05月16日 15:40\n请你模拟系统设置的角色") == ""


def test_extract_pairs_from_astrbot_jsonl_filters_noise():
    raw = (
        '{"content": ['
        '{"role": "user", "content": [{"type": "text", "text": "[发送时间: 2026-05-16 14:38]\\n你好"}]},'
        '{"role": "assistant", "content": [{"type": "text", "text": "你好呀"}]},'
        '{"role": "user", "content": [{"type": "text", "text": "当前时间是: 2026年05月16日 15:40\\n请你模拟系统设置的角色"}]},'
        '{"role": "assistant", "content": [{"type": "text", "text": "这条不该导入"}]}'
        ']}\n'
    )
    assert _extract_pairs_from_astrbot_jsonl(raw) == [("你好", "你好呀")]


def test_format_digest_pairs_marks_user_perspective():
    text = format_digest_pairs([("我今天拿到了 offer", "太好了")])
    assert "对方(用户)说: 我今天拿到了 offer" in text
    assert "我(AI)回应: 太好了" in text


# ===========================================================================
# Speaker-decoration helper — group-chat "不认人" regression suite.
#
# Without the per-message speaker prefix the digest LLM has no way to
# tell apart multiple users who all sit in the same ``对方(用户)`` slot
# of ``format_digest_pairs``, and ends up cross-attributing facts
# (the user's friend's bot doing this in group chats was the bug that
# prompted this fix). These tests pin the contract so the enrichment
# stays cohesive (a single helper) and zero-impact on private chats.
# ===========================================================================
def test_decorate_user_msg_with_speaker_basic():
    assert (
        decorate_user_msg_with_speaker("我今天拿到了 offer", "小明")
        == "[小明] 我今天拿到了 offer"
    )


def test_decorate_user_msg_with_speaker_none_speaker_is_noop():
    # Private chats / events without speaker info must pass through
    # bit-for-bit so existing summary text doesn't suddenly change shape.
    assert decorate_user_msg_with_speaker("hello", None) == "hello"
    assert decorate_user_msg_with_speaker("hello", "") == "hello"
    assert decorate_user_msg_with_speaker("hello", "   ") == "hello"


def test_decorate_user_msg_with_speaker_is_idempotent():
    # If the same buffer entry is somehow decorated twice (e.g. caller
    # passes already-tagged text) we must not nest prefixes — the LLM
    # would parse "[小明] [小明]" as confused metadata.
    first = decorate_user_msg_with_speaker("hi", "小明")
    second = decorate_user_msg_with_speaker(first, "小明")
    assert first == second == "[小明] hi"


def test_decorate_user_msg_with_speaker_strips_speaker_whitespace():
    assert (
        decorate_user_msg_with_speaker("hi", "  小明  ")
        == "[小明] hi"
    )


def test_decorate_user_msg_with_speaker_non_string_text_passes_through():
    # Defensive: a None / int slipping in (e.g. corrupt record) should
    # not raise inside the helper — callers above us already drop empty
    # text rows, so we only protect against type surprises here.
    assert decorate_user_msg_with_speaker(None, "小明") is None  # type: ignore[arg-type]


def test_convert_platform_history_default_does_not_decorate():
    # Backwards compat: existing private-chat call sites (no
    # ``include_speakers`` kwarg) must produce exactly the old payload
    # so we don't disturb summaries on platforms / configurations that
    # don't care about speakers.
    records = [
        SimpleNamespace(content="hi from alice", sender_name="alice", sender_id="1"),
        SimpleNamespace(content="hi from bob", sender_name="bob", sender_id="2"),
    ]
    out = _convert_platform_history(records)
    assert out == [
        {"role": "user", "content": "hi from alice"},
        {"role": "user", "content": "hi from bob"},
    ]


def test_convert_platform_history_with_speakers_tags_user_role_only():
    # Group chat path: distinct senders get their name prefixed so the
    # digest LLM can attribute statements correctly. Assistant messages
    # (detected via sender_name in ``_ASSISTANT_SENDER_HINTS``) keep
    # their original content because there is only one "I" in the
    # conversation and ``format_digest_pairs`` already labels them.
    records = [
        SimpleNamespace(content="我今天拿到 offer 了", sender_name="小明", sender_id="1"),
        SimpleNamespace(content="恭喜！", sender_name="bot", sender_id="0"),
        SimpleNamespace(content="我家狗子刚没了", sender_name="小红", sender_id="2"),
    ]
    out = _convert_platform_history(records, include_speakers=True)
    assert out == [
        {"role": "user", "content": "[小明] 我今天拿到 offer 了"},
        {"role": "assistant", "content": "恭喜！"},
        {"role": "user", "content": "[小红] 我家狗子刚没了"},
    ]


def test_convert_platform_history_with_speakers_skips_empty_name():
    # If a record has no sender_name we cannot enrich — leave the text
    # alone rather than emit ``[] ...`` garbage.
    records = [
        SimpleNamespace(content="hello", sender_name="", sender_id=""),
    ]
    out = _convert_platform_history(records, include_speakers=True)
    assert out == [{"role": "user", "content": "hello"}]


def test_event_is_group_chat_uses_astrbot_method():
    # When the AstrBot ``is_private_chat`` method is present we trust
    # its boolean answer — that's the canonical signal.
    private = SimpleNamespace(is_private_chat=lambda: True, unified_msg_origin="x")
    group = SimpleNamespace(is_private_chat=lambda: False, unified_msg_origin="x")
    assert _event_is_group_chat(private) is False
    assert _event_is_group_chat(group) is True


def test_event_is_group_chat_falls_back_to_umo_substring():
    # Some test stubs / unusual adapters don't expose ``is_private_chat``
    # — we should still classify based on the unified_msg_origin so
    # speaker enrichment lights up for obvious group umos.
    group = SimpleNamespace(unified_msg_origin="qq:GroupMessage:12345")
    private = SimpleNamespace(unified_msg_origin="qq:FriendMessage:7890")
    unknown = SimpleNamespace(unified_msg_origin="weird:adapter:abc")
    assert _event_is_group_chat(group) is True
    assert _event_is_group_chat(private) is False
    # Unknown shapes default to ``False`` so we don't accidentally
    # decorate private chats whose umo we can't parse.
    assert _event_is_group_chat(unknown) is False


def test_event_is_group_chat_swallows_exceptions():
    # A misbehaving ``is_private_chat`` (raises) should fall back to
    # ``unified_msg_origin`` rather than propagate up to the hook
    # pipeline — wrong attribution beats a crashed handler.
    def boom():
        raise RuntimeError("adapter exploded")

    event = SimpleNamespace(
        is_private_chat=boom,
        unified_msg_origin="qq:GroupMessage:12345",
    )
    assert _event_is_group_chat(event) is True


# ===========================================================================
# /memory list
# ===========================================================================
@ASYNCIO
async def test_list_empty_session(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        results = await _collect(obj.cmd_memory_list(FakeEvent(), 10))
        assert len(results) == 1
        assert "还没有任何记忆" in results[0].text
    finally:
        await db.close()


@ASYNCIO
async def test_list_with_buckets(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        await obj.manager.create_simple("qq:GroupMessage:12345", "alpha")
        await obj.manager.create_simple("qq:GroupMessage:12345", "beta")
        results = await _collect(obj.cmd_memory_list(FakeEvent(), 10))
        assert len(results) == 1
        out = results[0].text
        assert "alpha" in out
        assert "beta" in out
        # Score column present
        assert "score=" in out
    finally:
        await db.close()


@ASYNCIO
async def test_list_clamps_limit(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        for i in range(80):
            await obj.manager.create_simple("qq:GroupMessage:12345", f"x{i}")
        # Ask for 9999 — must be capped at LIST_HARD_CAP=50.
        results = await _collect(obj.cmd_memory_list(FakeEvent(), 9999))
        text = results[0].text
        # Header should say count is 50/80
        assert "50/80" in text
    finally:
        await db.close()


@ASYNCIO
async def test_list_uninitialised_returns_friendly_message():
    obj = FakeAssembled()
    results = await _collect(obj.cmd_memory_list(FakeEvent(), 10))
    assert "未初始化" in results[0].text


# ===========================================================================
# /memory search
# ===========================================================================
@ASYNCIO
async def test_search_returns_hit(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        await obj.manager.create_simple(
            "qq:GroupMessage:12345", "拿到实习 offer", name="实习offer"
        )
        await obj.manager.create_simple(
            "qq:GroupMessage:12345", "无关日常对话", name="日常"
        )
        results = await _collect(obj.cmd_memory_search(FakeEvent(), "实习"))
        out = results[0].text
        assert "实习" in out
    finally:
        await db.close()


@ASYNCIO
async def test_search_empty_query_shows_usage(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        results = await _collect(obj.cmd_memory_search(FakeEvent(), ""))
        assert "用法" in results[0].text
    finally:
        await db.close()


# ===========================================================================
# /memory pin
# ===========================================================================
@ASYNCIO
async def test_pin_toggles_state(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        b = await obj.manager.create_simple("qq:GroupMessage:12345", "x")
        # First call pins.
        results = await _collect(obj.cmd_memory_pin(FakeEvent(), b.id))
        assert "📌" in results[0].text or "钉选" in results[0].text
        loaded = await obj.manager.get("qq:GroupMessage:12345", b.id)
        assert loaded is not None
        assert loaded.pinned is True
        # Second call unpins.
        results = await _collect(obj.cmd_memory_pin(FakeEvent(), b.id))
        assert "取消" in results[0].text
        loaded = await obj.manager.get("qq:GroupMessage:12345", b.id)
        assert loaded is not None
        assert loaded.pinned is False
    finally:
        await db.close()


@ASYNCIO
async def test_pin_unknown_id(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        results = await _collect(obj.cmd_memory_pin(FakeEvent(), "no-such-id"))
        assert "未找到" in results[0].text
    finally:
        await db.close()


# ===========================================================================
# /memory forget
# ===========================================================================
@ASYNCIO
async def test_forget_marks_resolved(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        b = await obj.manager.create_simple("qq:GroupMessage:12345", "x")
        results = await _collect(obj.cmd_memory_forget(FakeEvent(), b.id))
        assert "沉底" in results[0].text
        loaded = await obj.manager.get("qq:GroupMessage:12345", b.id)
        assert loaded is not None
        assert loaded.resolved is True
    finally:
        await db.close()


# ===========================================================================
# /memory delete (two-step confirm)
# ===========================================================================
@ASYNCIO
async def test_delete_requires_confirmation(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        b = await obj.manager.create_simple("qq:GroupMessage:12345", "x")
        # First call: warning, no deletion.
        results = await _collect(obj.cmd_memory_delete(FakeEvent(), b.id, ""))
        assert "永久删除" in results[0].text
        assert "confirm" in results[0].text
        loaded = await obj.manager.get("qq:GroupMessage:12345", b.id)
        assert loaded is not None  # NOT deleted yet
    finally:
        await db.close()


@ASYNCIO
async def test_delete_confirmed_executes(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        _pending.clear()  # ensure clean test state
        b = await obj.manager.create_simple("qq:GroupMessage:12345", "x")
        await _collect(obj.cmd_memory_delete(FakeEvent(), b.id, ""))  # primes pending
        results = await _collect(
            obj.cmd_memory_delete(FakeEvent(), b.id, "confirm")
        )
        assert "永久删除" in results[0].text or "已永久删除" in results[0].text
        loaded = await obj.manager.get("qq:GroupMessage:12345", b.id)
        assert loaded is None
    finally:
        await db.close()


@ASYNCIO
async def test_delete_confirm_without_priming_fails(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        _pending.clear()
        b = await obj.manager.create_simple("qq:GroupMessage:12345", "x")
        # Skip the priming step — go straight to confirm.
        results = await _collect(
            obj.cmd_memory_delete(FakeEvent(), b.id, "confirm")
        )
        assert "已过期" in results[0].text or "不匹配" in results[0].text
        loaded = await obj.manager.get("qq:GroupMessage:12345", b.id)
        assert loaded is not None  # not deleted
    finally:
        await db.close()


@ASYNCIO
async def test_delete_unknown_id(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        results = await _collect(
            obj.cmd_memory_delete(FakeEvent(), "no-such-id", "")
        )
        assert "未找到" in results[0].text
    finally:
        await db.close()


@ASYNCIO
async def test_delete_session_isolation(tmp_path: Path):
    """A delete priming in session A must not be confirm-able from session B."""
    db, obj = await _open(tmp_path)
    try:
        _pending.clear()
        b = await obj.manager.create_simple("session-A", "alpha")

        # Prime in session A.
        await _collect(
            obj.cmd_memory_delete(
                FakeEvent(unified_msg_origin="session-A"), b.id, ""
            )
        )
        # Confirm from session B — must miss because pending is keyed by session.
        # First, session B can't even see the bucket.
        results = await _collect(
            obj.cmd_memory_delete(
                FakeEvent(unified_msg_origin="session-B"),
                b.id,
                "confirm",
            )
        )
        assert "未找到" in results[0].text
        # A's bucket is intact.
        loaded = await obj.manager.get("session-A", b.id)
        assert loaded is not None
    finally:
        await db.close()


# ===========================================================================
# /memory clear (admin + two-step)
# ===========================================================================
@ASYNCIO
async def test_clear_requires_admin(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        await obj.manager.create_simple("qq:GroupMessage:12345", "x")
        # Non-admin event.
        results = await _collect(obj.cmd_memory_clear(FakeEvent(is_admin=False), ""))
        assert "管理员" in results[0].text
        # Bucket untouched.
        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        assert len(buckets) == 1
    finally:
        await db.close()


@ASYNCIO
async def test_clear_two_step_admin(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        _pending.clear()
        for i in range(3):
            await obj.manager.create_simple("qq:GroupMessage:12345", f"x{i}")

        # First admin call primes pending, no deletion.
        results = await _collect(
            obj.cmd_memory_clear(FakeEvent(is_admin=True), "")
        )
        assert "全部" in results[0].text
        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        assert len(buckets) == 3

        # Confirm.
        results = await _collect(
            obj.cmd_memory_clear(FakeEvent(is_admin=True), "confirm")
        )
        assert "已清空" in results[0].text
        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        assert buckets == []
    finally:
        await db.close()


@ASYNCIO
async def test_clear_empty_session_returns_friendly(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        results = await _collect(
            obj.cmd_memory_clear(FakeEvent(is_admin=True), "")
        )
        assert "没有需要清除" in results[0].text
    finally:
        await db.close()


# ===========================================================================
# /memory stats
# ===========================================================================
@ASYNCIO
async def test_stats_returns_counts(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        await obj.manager.create_simple("qq:GroupMessage:12345", "x")
        await obj.manager.create_simple(
            "qq:GroupMessage:12345", "y", pinned=True
        )
        await obj.manager.create_simple(
            "qq:GroupMessage:12345", "z", bucket_type="feel"
        )
        results = await _collect(obj.cmd_memory_stats(FakeEvent()))
        out = results[0].text
        assert "动态: 1" in out
        assert "钉选/永久: 1" in out
        assert "感受 (feel): 1" in out
        assert "合计: 3" in out
    finally:
        await db.close()


@ASYNCIO
async def test_stats_shows_every_n_turns_counter(tmp_path: Path):
    """`/memory stats` must surface the current auto-record counter so users
    can see how close they are to the next auto-summary without grepping logs."""
    db, obj = await _open(tmp_path)
    try:
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "every_n_turns",
            "auto_record_every_n_turns": 30,
        }
        # Seed the persisted counter at 7 — `/memory stats` reads from
        # SQLite via the manager (schema v2 ``session_state`` table).
        await obj.manager.set_auto_record_counter("qq:GroupMessage:12345", 7)

        results = await _collect(obj.cmd_memory_stats(FakeEvent()))
        out = results[0].text
        assert "every_n_turns 7/30" in out
        assert "还差 23 轮" in out
    finally:
        await db.close()


@ASYNCIO
async def test_stats_shows_disabled_mode(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        obj.config = {
            "auto_record_enabled": True,
            "auto_record_mode": "disabled",
        }
        results = await _collect(obj.cmd_memory_stats(FakeEvent()))
        assert "disabled" in results[0].text
    finally:
        await db.close()


@ASYNCIO
async def test_stats_shows_master_toggle_off(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        obj.config = {"auto_record_enabled": False}
        results = await _collect(obj.cmd_memory_stats(FakeEvent()))
        assert "已关闭" in results[0].text
    finally:
        await db.close()


# ===========================================================================
# /memory import_astrbot
# ===========================================================================
@ASYNCIO
async def test_import_astrbot_jsonl_creates_memories(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        sample = tmp_path / "sample.jsonl"
        sample.write_text(
            '{"content": ['
            '{"role": "user", "content": [{"type": "text", "text": "[发送时间: 2026-05-16 14:38]\\n我今天拿到了 offer"}]},'
            '{"role": "assistant", "content": [{"type": "text", "text": "太好了，恭喜你"}]}'
            ']}\n',
            encoding="utf-8",
        )

        results = await _collect(
            obj.cmd_memory_import_astrbot(FakeEvent(), str(sample), 10)
        )
        assert len(results) == 2
        assert "正在从 AstrBot 历史中导入" in results[0].text
        assert "提取了" in results[1].text

        buckets = await obj.manager.list_by_session("qq:GroupMessage:12345")
        assert buckets
        assert "offer" in buckets[0].content
    finally:
        await db.close()


@ASYNCIO
async def test_import_astrbot_requires_jsonl_extension(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        sample = tmp_path / "sample.json"
        sample.write_text("{}", encoding="utf-8")
        results = await _collect(
            obj.cmd_memory_import_astrbot(FakeEvent(), str(sample), 10)
        )
        assert "只支持 AstrBot 导出的 .jsonl" in results[0].text
    finally:
        await db.close()


# ===========================================================================
# /memory help
# ===========================================================================
@ASYNCIO
async def test_help_lists_subcommands(tmp_path: Path):
    db, obj = await _open(tmp_path)
    try:
        results = await _collect(obj.cmd_memory_help(FakeEvent()))
        out = results[0].text
        for sub in ("list", "search", "pin", "forget", "delete", "clear", "stats"):
            assert f"/memory {sub}" in out
    finally:
        await db.close()
