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
    # ``None`` means "no is_private_chat method bound" — used to
    # exercise the umo-substring fallback path in the resolver.
    _is_private: bool | None = False

    def get_sender_id(self) -> str:
        return self.sender_id

    def is_private_chat(self) -> bool | None:
        return self._is_private


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

    def __init__(
        self,
        *,
        scope_mode: str | None = "conversation",
        context=None,
        unify_groups_into_user: str | None = None,
    ):
        self.config: dict = {"scope_mode": scope_mode} if scope_mode else {}
        if unify_groups_into_user is not None:
            self.config["unify_groups_into_user"] = unify_groups_into_user
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


# ---------------------------------------------------------------------------
# Hybrid mode — private uses user semantics, group uses origin semantics
# ---------------------------------------------------------------------------
@ASYNCIO
async def test_hybrid_private_chat_uses_user_semantics():
    """In a private chat, hybrid mode should behave like ``user`` so all
    of that user's private windows share one memory pool."""
    plugin = FakePlugin(scope_mode="hybrid", context=FakeContext(None))
    resolver = SessionResolver(plugin)

    sid = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:FriendMessage:111",
            sender_id="kk",
            _is_private=True,
        )
    )
    assert sid == "user:kk"


@ASYNCIO
async def test_hybrid_group_chat_uses_origin_semantics():
    """In a group chat, hybrid mode should behave like ``origin`` so
    the whole group shares one memory pool instead of fragmenting by
    speaker."""
    plugin = FakePlugin(scope_mode="hybrid", context=FakeContext(None))
    resolver = SessionResolver(plugin)

    sid = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:GroupMessage:777",
            sender_id="kk",
            _is_private=False,
        )
    )
    assert sid == "aiocqhttp:GroupMessage:777"


@ASYNCIO
async def test_hybrid_group_shared_across_speakers():
    """Two different speakers in the same group → same session_id.
    This is the key difference from ``user`` mode (which would give
    each speaker their own ``user:`` pool)."""
    plugin = FakePlugin(scope_mode="hybrid", context=FakeContext(None))
    resolver = SessionResolver(plugin)

    sid_kk = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:GroupMessage:777",
            sender_id="kk",
            _is_private=False,
        )
    )
    sid_other = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:GroupMessage:777",
            sender_id="someone-else",
            _is_private=False,
        )
    )
    assert sid_kk == sid_other == "aiocqhttp:GroupMessage:777"


@ASYNCIO
async def test_hybrid_private_shared_across_windows():
    """Same private user → same ``user:`` session whatever the
    conversation cid is. This is the key difference from
    ``conversation`` mode."""
    plugin = FakePlugin(scope_mode="hybrid", context=FakeContext(None))
    resolver = SessionResolver(plugin)

    sid_1 = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:FriendMessage:111",
            sender_id="kk",
            _is_private=True,
        )
    )
    sid_2 = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:FriendMessage:111",
            sender_id="kk",
            _is_private=True,
        )
    )
    assert sid_1 == sid_2 == "user:kk"


@ASYNCIO
async def test_hybrid_falls_back_to_umo_when_is_private_missing():
    """Test stubs / exotic adapters without ``is_private_chat()`` —
    detection must still work via the umo substring fallback."""

    @dataclass
    class NoMethodEvent:
        unified_msg_origin: str = "aiocqhttp:FriendMessage:222"
        sender_id: str = "alice"

        def get_sender_id(self) -> str:
            return self.sender_id

    plugin = FakePlugin(scope_mode="hybrid", context=FakeContext(None))
    resolver = SessionResolver(plugin)

    sid_priv = await resolver.resolve(NoMethodEvent())
    assert sid_priv == "user:alice"

    sid_grp = await resolver.resolve(
        NoMethodEvent(unified_msg_origin="aiocqhttp:GroupMessage:42")
    )
    assert sid_grp == "aiocqhttp:GroupMessage:42"


@ASYNCIO
async def test_hybrid_falls_back_to_origin_when_private_but_no_sender():
    """Even in private mode, missing sender_id falls back to origin
    so memory recall keeps working."""
    plugin = FakePlugin(scope_mode="hybrid", context=FakeContext(None))
    resolver = SessionResolver(plugin)
    sid = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:FriendMessage:111",
            sender_id="",
            _is_private=True,
        )
    )
    assert sid == "aiocqhttp:FriendMessage:111"


def test_hybrid_is_a_valid_mode():
    assert "hybrid" in VALID_SCOPE_MODES


# ---------------------------------------------------------------------------
# unify_groups_into_user — group activity funnels into owner's user pool
# ---------------------------------------------------------------------------
@ASYNCIO
async def test_unify_redirects_group_event_to_owner_user_pool():
    """When ``unify_groups_into_user`` is set, group events resolve to
    ``user:{owner_id}`` so group activity merges into the owner's
    private memory pool."""
    plugin = FakePlugin(
        scope_mode="hybrid",
        context=FakeContext(None),
        unify_groups_into_user="2652497429",
    )
    resolver = SessionResolver(plugin)

    sid = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:GroupMessage:777",
            sender_id="some-group-member",
            _is_private=False,
        )
    )
    assert sid == "user:2652497429"


@ASYNCIO
async def test_unify_does_not_affect_private_events():
    """Private chats stay on their own ``user:{sender_id}`` pool — we
    don't want random other users' private chats to bleed into the
    owner's pool."""
    plugin = FakePlugin(
        scope_mode="hybrid",
        context=FakeContext(None),
        unify_groups_into_user="2652497429",
    )
    resolver = SessionResolver(plugin)

    sid_owner = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:FriendMessage:2652497429",
            sender_id="2652497429",
            _is_private=True,
        )
    )
    # Owner's private chat still resolves to their own user pool —
    # which happens to equal the unify target. That's the whole
    # point: private + group converge on the same key.
    assert sid_owner == "user:2652497429"

    sid_other = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:FriendMessage:other",
            sender_id="other-user",
            _is_private=True,
        )
    )
    # Some other user's private chat is unaffected — they keep their
    # own pool, not bleeding into the owner's.
    assert sid_other == "user:other-user"


@ASYNCIO
async def test_unify_works_for_user_mode_too():
    """In ``user`` mode, every group speaker would otherwise resolve
    to their own ``user:{speaker_id}``. With unify, they all merge
    into the owner's pool."""
    plugin = FakePlugin(
        scope_mode="user",
        context=FakeContext(None),
        unify_groups_into_user="2652497429",
    )
    resolver = SessionResolver(plugin)

    sid_kk = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:GroupMessage:777",
            sender_id="2652497429",
            _is_private=False,
        )
    )
    sid_friend = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:GroupMessage:777",
            sender_id="some-friend",
            _is_private=False,
        )
    )
    assert sid_kk == sid_friend == "user:2652497429"


@ASYNCIO
async def test_unify_works_for_origin_mode_too():
    plugin = FakePlugin(
        scope_mode="origin",
        context=FakeContext(None),
        unify_groups_into_user="2652497429",
    )
    resolver = SessionResolver(plugin)

    sid = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:GroupMessage:777",
            sender_id="any",
            _is_private=False,
        )
    )
    assert sid == "user:2652497429"


@ASYNCIO
async def test_unify_ignored_in_conversation_mode():
    """``conversation`` mode is per-window-isolated by design.
    Honouring unify would silently merge windows that the user
    explicitly asked to keep separate, so we skip it there."""
    plugin = FakePlugin(
        scope_mode="conversation",
        context=FakeContext(FakeConversationManager({"qq:GroupMessage:777": "cid-G"})),
        unify_groups_into_user="2652497429",
    )
    resolver = SessionResolver(plugin)
    sid = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="qq:GroupMessage:777",
            sender_id="any",
            _is_private=False,
        )
    )
    assert sid == "conv:cid-G"


@ASYNCIO
async def test_unify_empty_string_is_treated_as_disabled():
    """Empty config value = toggle disabled = base hybrid behaviour."""
    plugin = FakePlugin(
        scope_mode="hybrid",
        context=FakeContext(None),
        unify_groups_into_user="   ",  # whitespace counts as empty
    )
    resolver = SessionResolver(plugin)
    sid = await resolver.resolve(
        FakeEvent(
            unified_msg_origin="aiocqhttp:GroupMessage:777",
            sender_id="any",
            _is_private=False,
        )
    )
    assert sid == "aiocqhttp:GroupMessage:777"


# ---------------------------------------------------------------------------
# resolve_counter_key — per-conversation cadence regardless of scope_mode
# ---------------------------------------------------------------------------
@ASYNCIO
async def test_counter_key_uses_cid_when_available():
    """A cid produces a ``conv:`` counter key whatever ``scope_mode`` is."""
    cm = FakeConversationManager(mapping={"qq:Private:111": "cid-private"})
    plugin = FakePlugin(scope_mode="user", context=FakeContext(cm))
    resolver = SessionResolver(plugin)

    key = await resolver.resolve_counter_key(
        FakeEvent(unified_msg_origin="qq:Private:111")
    )
    assert key == "conv:cid-private"


@ASYNCIO
async def test_counter_key_falls_back_to_origin_when_no_cid():
    """No cid (e.g. adapter that doesn't track group conversations)
    → origin-scoped key, so private vs. group are still separate."""
    cm = FakeConversationManager(mapping={"qq:GroupMessage:42": None})
    plugin = FakePlugin(scope_mode="user", context=FakeContext(cm))
    resolver = SessionResolver(plugin)

    key = await resolver.resolve_counter_key(
        FakeEvent(unified_msg_origin="qq:GroupMessage:42")
    )
    assert key == "origin:qq:GroupMessage:42"


@ASYNCIO
async def test_counter_key_falls_back_when_manager_crashes():
    plugin = FakePlugin(
        scope_mode="user",
        context=FakeContext(CrashConversationManager()),
    )
    resolver = SessionResolver(plugin)
    key = await resolver.resolve_counter_key(
        FakeEvent(unified_msg_origin="qq:GroupMessage:99")
    )
    assert key == "origin:qq:GroupMessage:99"


@ASYNCIO
async def test_counter_key_distinct_per_cid_under_user_scope():
    """Two different AstrBot conversations under the same QQ user must
    get **different** counter keys even when ``scope_mode=user`` shares
    their memory pool.
    """
    cm = FakeConversationManager()
    plugin = FakePlugin(scope_mode="user", context=FakeContext(cm))
    resolver = SessionResolver(plugin)

    cm.mapping["qq:Private:111"] = "cid-window-1"
    k1 = await resolver.resolve_counter_key(
        FakeEvent(unified_msg_origin="qq:Private:111")
    )

    cm.mapping["qq:Private:111"] = "cid-window-2"
    k2 = await resolver.resolve_counter_key(
        FakeEvent(unified_msg_origin="qq:Private:111")
    )

    # Memory side: both events still resolve to ``user:user-A``.
    mem1 = await resolver.resolve(
        FakeEvent(unified_msg_origin="qq:Private:111", sender_id="user-A")
    )
    assert mem1 == "user:user-A"

    # Counter side: two distinct keys, so each window has its own cadence.
    assert k1 == "conv:cid-window-1"
    assert k2 == "conv:cid-window-2"
    assert k1 != k2


@ASYNCIO
async def test_counter_key_separates_private_vs_group_origin_fallback():
    """When neither private nor group has a tracked cid (extreme case)
    the origin fallback still keeps them apart — by origin string."""
    cm = FakeConversationManager()  # no entries
    plugin = FakePlugin(scope_mode="user", context=FakeContext(cm))
    resolver = SessionResolver(plugin)

    priv = await resolver.resolve_counter_key(
        FakeEvent(unified_msg_origin="qq:Private:111")
    )
    grp = await resolver.resolve_counter_key(
        FakeEvent(unified_msg_origin="qq:GroupMessage:777")
    )
    assert priv == "origin:qq:Private:111"
    assert grp == "origin:qq:GroupMessage:777"
    assert priv != grp
