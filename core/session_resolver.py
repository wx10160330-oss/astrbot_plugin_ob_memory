"""Session-id resolution.

Decides what string we use as the storage isolation key for an
``AstrMessageEvent``. Four modes are supported:

- ``conversation`` (default): one record-pool per AstrBot conversation
  window. The user's ``/new`` command starts a fresh memory pool. This
  is the recommended setting for AI-companion-style use because it
  matches how users intuitively think of "this AI's memory of this
  chat". Implementation: read the current cid via
  ``context.conversation_manager.get_curr_conversation_id``.
- ``user``: same memory pool follows the user across all their windows
  and across platforms (QQ private + Telegram private + group, etc).
  For users who want one cohesive AI companion regardless of where
  they're chatting from. Implementation: ``f"user:{sender_id}"``.
- ``origin``: shared by everyone in the same ``unified_msg_origin``.
  Useful for "group-wide shared memory" scenarios — bot remembers
  events that happened in a group, every member sees the same recall.
- ``hybrid``: private chats use ``user`` semantics (one shared pool
  per user across all their private windows), group chats use
  ``origin`` semantics (one shared pool per group). The natural
  default for users who want "cross-window continuity in private but
  group-wide shared recall in groups" without the ``user``-mode
  side effect of fragmenting groups by speaker.

The resolver is intentionally **stateless** beyond ``self.plugin``:
config is re-read on every call so a Dashboard / config UI change
takes effect on the next message, not after a restart.

It is also **fail-safe**: any exception, missing attribute, or
unrecognised mode falls back to ``event.unified_msg_origin`` and logs
the issue once. Memory recall NEVER breaks because of resolver errors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

    from ..main import MemoryPlugin


logger = logging.getLogger("astrbot_plugin_ob_memory.session")


VALID_SCOPE_MODES: tuple[str, ...] = (
    "conversation",
    "user",
    "origin",
    "hybrid",
)
"""Accepted ``session.scope_mode`` values.

The order doubles as a documentation-style "default first" hint.
"""

DEFAULT_SCOPE_MODE: str = "conversation"


def _origin_of(event: AstrMessageEvent) -> str:
    """Best-effort fallback when the configured mode can't produce a key."""
    sid = getattr(event, "unified_msg_origin", None)
    if not sid:
        sid = getattr(event, "session_id", None)
    return str(sid or "unknown")


class SessionResolver:
    """Maps an event to the storage key used for isolation.

    The resolver is bound to a ``MemoryPlugin`` so it can read live
    config from ``self.plugin.config``. Tests can pass a stand-in
    object that exposes ``config`` and an optional ``context``.
    """

    def __init__(self, plugin: MemoryPlugin):
        self.plugin = plugin
        # We log "unknown scope_mode" once per process to avoid log
        # spam if the user has misconfigured the value. The set holds
        # the bad strings already warned about.
        self._warned_unknown_modes: set[str] = set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mode(self) -> str:
        """Read ``scope_mode`` from live config.

        Tolerates a missing key, a non-string value, or a value not in
        :data:`VALID_SCOPE_MODES`.
        """
        cfg = getattr(self.plugin, "config", None)
        if not isinstance(cfg, dict):
            return DEFAULT_SCOPE_MODE

        raw = cfg.get("scope_mode")
        if not isinstance(raw, str):
            return DEFAULT_SCOPE_MODE

        mode = raw.strip().lower()
        if mode in VALID_SCOPE_MODES:
            return mode

        # Unknown mode — fall back to default; warn once.
        if mode and mode not in self._warned_unknown_modes:
            self._warned_unknown_modes.add(mode)
            logger.warning(
                "unknown scope_mode %r; falling back to %r. valid options: %s",
                mode,
                DEFAULT_SCOPE_MODE,
                ", ".join(VALID_SCOPE_MODES),
            )
        return DEFAULT_SCOPE_MODE

    async def _conversation_id(self, event: AstrMessageEvent) -> str | None:
        """Look up the AstrBot conversation id (cid) for this event.

        Returns ``None`` when there is no active conversation or the
        conversation manager is unavailable (tests, early init, etc).
        """
        ctx = getattr(self.plugin, "context", None)
        if ctx is None:
            return None
        mgr = getattr(ctx, "conversation_manager", None)
        if mgr is None:
            return None
        umo = getattr(event, "unified_msg_origin", None)
        if not umo:
            return None
        try:
            cid = await mgr.get_curr_conversation_id(str(umo))
        except Exception as e:
            logger.debug("conversation_manager lookup failed: %s", e)
            return None
        return str(cid) if cid else None

    def _is_private_chat(self, event: AstrMessageEvent) -> bool:
        """Decide whether this event is a private/direct message.

        Prefers AstrBot's :meth:`is_private_chat` when available; falls
        back to a substring check against ``unified_msg_origin`` so
        test stubs and exotic adapters still work.

        Any unrecognised origin defaults to ``False`` (treated as
        group) so ``hybrid`` mode keeps group-style sharing semantics
        in the ambiguous case rather than accidentally fragmenting a
        group pool by sender.
        """
        getter = getattr(event, "is_private_chat", None)
        if callable(getter):
            try:
                value = getter()
            except Exception as e:
                logger.debug("is_private_chat failed: %s", e)
                value = None
            if isinstance(value, bool):
                return value
        umo = getattr(event, "unified_msg_origin", "") or ""
        umo_str = str(umo)
        if "FriendMessage" in umo_str or ":Private:" in umo_str or ":Friend:" in umo_str:
            return True
        if "GroupMessage" in umo_str or ":Group:" in umo_str:
            return False
        return False

    def _sender_id(self, event: AstrMessageEvent) -> str | None:
        """Read the sender id, accepting either method or attribute."""
        getter = getattr(event, "get_sender_id", None)
        if callable(getter):
            try:
                value = getter()
            except Exception as e:
                logger.debug("get_sender_id failed: %s", e)
                return None
            return str(value).strip() or None
        # Attribute fallback for stub events used in tests.
        attr = getattr(event, "sender_id", None)
        if attr:
            return str(attr).strip() or None
        return None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    async def resolve(self, event: AstrMessageEvent) -> str:
        """Return the session_id string this event should map to.

        Never raises — on any failure, falls back to the unified message
        origin so memory recall keeps working.
        """
        try:
            mode = self._mode()
            if mode == "user":
                sender = self._sender_id(event)
                if sender:
                    return f"user:{sender}"
                return _origin_of(event)
            if mode == "conversation":
                cid = await self._conversation_id(event)
                if cid:
                    return f"conv:{cid}"
                return _origin_of(event)
            if mode == "hybrid":
                # Private chat → user semantics so all the user's
                # private windows share one pool. Group chat (or any
                # non-private event) → origin semantics so the whole
                # group shares one pool rather than fragmenting by
                # speaker.
                if self._is_private_chat(event):
                    sender = self._sender_id(event)
                    if sender:
                        return f"user:{sender}"
                return _origin_of(event)
            # ``origin`` mode (and the safety fallback)
            return _origin_of(event)
        except Exception as e:
            logger.warning("session resolve crashed; using origin fallback: %s", e)
            return _origin_of(event)

    async def resolve_counter_key(self, event: AstrMessageEvent) -> str:
        """Return the storage key used for the ``every_n_turns`` counter.

        Decoupled from :meth:`resolve` on purpose. The memory pool may
        be shared across conversations (``scope_mode = user``) but the
        user-facing summary cadence should still feel "per-window":

        - User opens a fresh AstrBot conversation (``/new``) → fresh
          counter, fresh summary cycle.
        - Group chat and private chat → distinct counters even if both
          write into the same ``user:`` memory pool.

        The key picks the most granular id that AstrBot can give us:

        1. ``conv:{cid}`` — when ``conversation_manager`` has a current
           conversation id for this event's UMO. This is the common
           path for private chats.
        2. ``origin:{umo}`` — fallback for adapters / chat types that
           don't expose a cid (e.g. some group adapters). Sharing one
           counter across the whole origin is still better than
           sharing one across the whole user, because at least private
           vs. group are kept apart.

        Always returns a non-empty string.
        """
        try:
            cid = await self._conversation_id(event)
            if cid:
                return f"conv:{cid}"
            return f"origin:{_origin_of(event)}"
        except Exception as e:
            logger.warning(
                "counter-key resolve crashed; using origin fallback: %s", e
            )
            return f"origin:{_origin_of(event)}"
