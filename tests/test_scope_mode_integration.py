"""Integration tests for scope_mode end-to-end through LLM tools.

Confirms the resolver actually changes which buckets get returned by
``record_memory`` / ``recall_memory`` when the configured scope_mode
flips between modes.

Also covers the spec-mandated invariant: switching scope_mode at
runtime does NOT migrate existing rows. A bucket written under one
scope is invisible under another, but switching back makes it visible
again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.core.memory_writer import MemoryWriter
from astrbot_plugin_ob_memory.core.search_service import SearchService
from astrbot_plugin_ob_memory.core.session_resolver import SessionResolver
from astrbot_plugin_ob_memory.core.tagger import Tagger

from astrbot_plugin_ob_memory.handlers.llm_tools import MemoryToolsMixin

from astrbot_plugin_ob_memory.storage import Database, apply_migrations

ASYNCIO = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------
@dataclass
class FakeEvent:
    unified_msg_origin: str = "qq:Private:user-A"

    def get_sender_id(self) -> str:
        return "user-A"


class FakeConvManager:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    async def get_curr_conversation_id(self, umo: str):
        return self.mapping.get(umo)


class FakeContext:
    def __init__(self, conv_mgr):
        self.conversation_manager = conv_mgr


class FakeLLMResponse:
    def __init__(self, text: str = ""):
        self.completion_text = text


class StubProvider:
    """Returns a default-shaped analyse JSON regardless of input."""

    async def text_chat(self, prompt=None, system_prompt=None, **kw):
        import json

        return FakeLLMResponse(
            json.dumps(
                {
                    "domain": ["未分类"],
                    "valence": 0.5,
                    "arousal": 0.3,
                    "tags": [],
                    "suggested_name": "",
                    "importance": 5,
                }
            )
        )


class FakeAssembled(MemoryToolsMixin):
    """Minimal stand-in that owns a SessionResolver."""

    def __init__(self, *, scope_mode: str, conv_mapping=None):
        self.config: dict = {"scope_mode": scope_mode}
        self.context = FakeContext(FakeConvManager(conv_mapping or {}))
        self.manager: MemoryManager | None = None
        self.search: SearchService | None = None
        self.embedding = None
        self.writer: MemoryWriter | None = None
        self.session_resolver = SessionResolver(self)


def _bind(method, instance):
    return method.__get__(instance, instance.__class__)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
async def _open(tmp_path: Path, scope_mode: str, conv_mapping=None):
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)

    obj = FakeAssembled(scope_mode=scope_mode, conv_mapping=conv_mapping)
    obj.manager = MemoryManager(db)
    obj.search = SearchService(obj.manager, embedding=None)
    obj.writer = MemoryWriter(
        obj.manager,
        tagger=Tagger(context=None, fixed_provider=StubProvider()),
        embedding=None,
        tagging_enabled=True,
        merge_enabled=False,
    )
    return db, obj


# ===========================================================================
# Tests
# ===========================================================================
@ASYNCIO
async def test_conversation_mode_isolates_per_cid(tmp_path: Path):
    """Two cids on the same origin → two separate memory pools."""
    db, obj = await _open(
        tmp_path,
        scope_mode="conversation",
        conv_mapping={"qq:Private:user-A": "cid-1"},
    )
    try:
        record = _bind(MemoryToolsMixin.record_memory, obj)
        recall = _bind(MemoryToolsMixin.recall_memory, obj)

        # Write under cid-1.
        await record(FakeEvent(), content="something from conversation 1")

        # Switch the active cid mid-flight (user opened a new chat).
        obj.context.conversation_manager.mapping["qq:Private:user-A"] = "cid-2"

        # Recall under cid-2 must NOT see cid-1's content.
        result = await recall(FakeEvent(), query="conversation 1")
        assert "没有找到" in result

        # Switch back — the old memory reappears.
        obj.context.conversation_manager.mapping["qq:Private:user-A"] = "cid-1"
        result_back = await recall(FakeEvent(), query="conversation 1")
        assert "conversation 1" in result_back
    finally:
        await db.close()


@ASYNCIO
async def test_user_mode_shares_across_cids(tmp_path: Path):
    """Same user across two cids → still one memory pool."""
    db, obj = await _open(
        tmp_path,
        scope_mode="user",
        conv_mapping={"qq:Private:user-A": "cid-1"},
    )
    try:
        record = _bind(MemoryToolsMixin.record_memory, obj)
        recall = _bind(MemoryToolsMixin.recall_memory, obj)

        await record(FakeEvent(), content="user-scoped fact")
        # Cid changes — under user mode this should NOT matter.
        obj.context.conversation_manager.mapping["qq:Private:user-A"] = "cid-2"
        result = await recall(FakeEvent(), query="user-scoped")
        assert "user-scoped" in result
    finally:
        await db.close()


@ASYNCIO
async def test_origin_mode_shares_across_users(tmp_path: Path):
    """In origin mode, two senders on the same origin share a pool."""
    db, obj = await _open(tmp_path, scope_mode="origin")
    try:
        record = _bind(MemoryToolsMixin.record_memory, obj)
        recall = _bind(MemoryToolsMixin.recall_memory, obj)

        # Sender A writes.
        await record(
            FakeEvent(unified_msg_origin="qq:GroupMessage:42"),
            content="shared group fact",
        )

        # Sender B (different sender_id) recalls — same origin, same pool.
        class BEvent(FakeEvent):
            unified_msg_origin = "qq:GroupMessage:42"

            def get_sender_id(self):
                return "user-B"

        result = await recall(BEvent(), query="shared group")
        assert "shared group" in result
    finally:
        await db.close()


@ASYNCIO
async def test_scope_mode_switch_does_not_migrate_data(tmp_path: Path):
    """The spec-mandated invariant: changing scope_mode at runtime does
    NOT migrate existing rows. A bucket written under conversation
    scope is invisible under user scope, and vice versa."""
    db, obj = await _open(
        tmp_path,
        scope_mode="conversation",
        conv_mapping={"qq:Private:user-A": "cid-1"},
    )
    try:
        record = _bind(MemoryToolsMixin.record_memory, obj)
        recall = _bind(MemoryToolsMixin.recall_memory, obj)

        # Write under conversation scope.
        await record(FakeEvent(), content="conv-scoped data")

        # Switch to user scope.
        obj.config["scope_mode"] = "user"

        # Same user, but different scope key → not visible.
        result = await recall(FakeEvent(), query="conv-scoped")
        assert "没有找到" in result

        # Switch back — visible again.
        obj.config["scope_mode"] = "conversation"
        result_back = await recall(FakeEvent(), query="conv-scoped")
        assert "conv-scoped" in result_back
    finally:
        await db.close()
