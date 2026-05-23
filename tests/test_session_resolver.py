"""Tests for ``core.session_resolver.SessionResolver``.

Three modes (``conversation`` / ``user`` / ``origin``), plus fallback
behaviour for malformed config and missing platform integrations.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from astrbot_plugin_ob_memory.core.session_resolver import (
    DEFAULT_SCOPE_MODE,
    VALID_SCOPE_MODES,
    SessionResolver,
)

ASYNCIO = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------
@dataclass
class FakeEvent:
    unified_msg_origin: str = "qq:GroupMessage:12345"
    sender_id: str = "user-A"

    def get_sender_id(self) -> str:
        return self.sender_id


class FakeConversationManager:
    def __init__(self, mapping: dict[str, str | None] | None = None):
        self.mapping = mapping or {}
        self.calls: list[str] = []

    async def get_curr_conversation_id(self, umo: str):
        self.calls.append(umo)
        return self.mapping.get(umo)


class CrashConversationManager:
    async def get_curr_conversation_id(self, umo: str):
        raise RuntimeError("conversation manager exploded")


class FakeContext:
    def __init__(self, conv_mgr=None):
        self.conversation_manager = conv_mgr


class FakePlugin:
    """Mimics the bits of MemoryPlugin the resolver touches."""

    def __init__(self, *, scope_mode: str | None = "conversation", context=None):
        self.config: dict = {"scope_mode": scope_mode} if scope_mode else {}
        self.context = context


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_default_mode_is_conversation():
    assert DEFAULT_SCOPE_MODE == "conversation"
    assert "conversation" in VALID_SCOPE_MODES
    assert "user" in VALID_SCOPE_MODES
    assert "origin" in VALID_SCOPE_MODES


# ---------------------------------------------------------------------------
# Conversation mode
# ---------------------------------------------------------------------------
@ASYNCIO
async def test_conversation_mode_uses_cid():
    cm = FakeConversationManager(mapping={"qq:GroupMessage:12345": "cid-abc-1"})
    plugin = FakePlugin(scope_mode="conversation", context=FakeContext(cm))
    resolver = SessionResolver(plugin)

    sid = await resolver.resolve(FakeEvent())
    assert sid == "conv:cid-abc-1"
    assert cm.calls == ["qq:GroupMessage:12345"]


@ASYNCIO
async def test_conversation_mode_fallback_when_no_cid():
    """No active conversation → fall back to origin so recall keeps working."""
    cm = FakeConversationManager(mapping={"qq:GroupMessage:12345": None})
    plugin = FakePlugin(scope_mode="conversation", context=FakeContext(cm))
    resolver = SessionResolver(plugin)

    sid = await resolver.resolve(FakeEvent())
    assert sid == "qq:GroupMessage:12345"


@ASYNCIO
async def test_conversation_mode_fallback_when_manager_missing():
    plugin = FakePlugin(scope_mode="conversation", context=FakeContext(None))
    resolver = SessionResolver(plugin)
    sid = await resolver.resolve(FakeEvent())
    assert sid == "qq:GroupMessage:12345"


@ASYNCIO
async def test_conversation_mode_fallback_when_manager_crashes():
    plugin = FakePlugin(
        scope_mode="conversation",
        context=FakeContext(CrashConversationManager()),
    )
    resolver = SessionResolver(plugin)
    sid = await resolver.resolve(FakeEvent())
    assert sid == "qq:GroupMessage:12345"


@ASYNCIO
async def test_two_distinct_cids_produce_distinct_session_ids():
    """Different cids on the same origin must isolate."""
    cm = FakeConversationManager()
    plugin = FakePlugin(scope_mode="conversation", context=FakeContext(cm))
    resolver = SessionResolver(plugin)

    cm.mapping["qq:GroupMessage:12345"] = "cid-a"
    sid_a = await resolver.resolve(FakeEvent())

    cm.mapping["qq:GroupMessage:12345"] = "cid-b"
    sid_b = await resolver.resolve(FakeEvent())

    assert sid_a != sid_b
    assert sid_a == "conv:cid-a"
    assert sid_b == "conv:cid-b"


# ---------------------------------------------------------------------------
# User mode
# ---------------------------------------------------------------------------
@ASYNCIO
async def test_user_mode_uses_sender_id():
    plugin = FakePlugin(scope_mode="user", context=FakeContext(None))
    resolver = SessionResolver(plugin)

    sid = await resolver.resolve(FakeEvent(sender_id="alice"))
    assert sid == "user:alice"


@ASYNCIO
async def test_user_mode_same_user_across_origins():
    """Same user on QQ private and Telegram private gets the same session."""
    plugin = FakePlugin(scope_mode="user", context=FakeContext(None))
    resolver = SessionResolver(plugin)

    sid_qq = await resolver.resolve(
        FakeEvent(unified_msg_origin="qq:Private:111", sender_id="alice")
    )
    sid_tg = await resolver.resolve(
        FakeEvent(unified_msg_origin="telegram:Private:111", sender_id="alice")
    )
    assert sid_qq == sid_tg == "user:alice"


@ASYNCIO
async def test_user_mode_falls_back_when_sender_missing():
    plugin = FakePlugin(scope_mode="user", context=FakeContext(None))
    resolver = SessionResolver(plugin)

    sid = await resolver.resolve(FakeEvent(sender_id=""))
    assert sid == "qq:GroupMessage:12345"


# ---------------------------------------------------------------------------
# Origin mode
# ---------------------------------------------------------------------------
@ASYNCIO
async def test_origin_mode_uses_unified_msg_origin():
    plugin = FakePlugin(scope_mode="origin", context=FakeContext(None))
    resolver = SessionResolver(plugin)

    sid = await resolver.resolve(FakeEvent(unified_msg_origin="qq:GroupMessage:9"))
    assert sid == "qq:GroupMessage:9"


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------
@ASYNCIO
async def test_unknown_mode_falls_back_to_default():
    plugin = FakePlugin(scope_mode="hyperdrive", context=FakeContext(None))
    resolver = SessionResolver(plugin)
    # Default is 'conversation' but no conv manager → falls to origin.
    sid = await resolver.resolve(FakeEvent())
    assert sid == "qq:GroupMessage:12345"


@ASYNCIO
async def test_missing_session_section_uses_default():
    plugin = FakePlugin(scope_mode=None, context=FakeContext(None))
    plugin.config = {}  # No session section at all
    resolver = SessionResolver(plugin)
    sid = await resolver.resolve(FakeEvent())
    # Default mode = conversation, no manager → origin fallback.
    assert sid == "qq:GroupMessage:12345"


@ASYNCIO
async def test_non_string_mode_uses_default():
    plugin = FakePlugin(scope_mode=None, context=FakeContext(None))
    plugin.config = {"scope_mode": 42}
    resolver = SessionResolver(plugin)
    sid = await resolver.resolve(FakeEvent())
    assert sid == "qq:GroupMessage:12345"


@ASYNCIO
async def test_unknown_mode_warned_only_once(caplog):
    import logging

    plugin = FakePlugin(scope_mode="bogus", context=FakeContext(None))
    resolver = SessionResolver(plugin)
    with caplog.at_level(logging.WARNING, logger="astrbot_plugin_ob_memory.session"):
        await resolver.resolve(FakeEvent())
        await resolver.resolve(FakeEvent())
        await resolver.resolve(FakeEvent())
    warnings = [r for r in caplog.records if "unknown scope_mode" in r.getMessage()]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Hot-reload
# ---------------------------------------------------------------------------
@ASYNCIO
async def test_mode_change_takes_effect_on_next_call():
    """No caching: flipping config between calls changes the next resolve."""
    cm = FakeConversationManager(mapping={"qq:GroupMessage:12345": "cid-1"})
    plugin = FakePlugin(scope_mode="origin", context=FakeContext(cm))
    resolver = SessionResolver(plugin)

    sid_origin = await resolver.resolve(FakeEvent())
    assert sid_origin == "qq:GroupMessage:12345"

    plugin.config["scope_mode"] = "conversation"
    sid_conv = await resolver.resolve(FakeEvent())
    assert sid_conv == "conv:cid-1"

    plugin.config["scope_mode"] = "user"
    sid_user = await resolver.resolve(FakeEvent())
    assert sid_user == "user:user-A"
