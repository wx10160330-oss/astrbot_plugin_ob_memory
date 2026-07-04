"""LLM-request and LLM-response hooks for the memory plugin.

These two hooks are where the memory system actually plugs into the
conversation:

- :meth:`on_llm_request` runs before every LLM call. It quietly
  searches and surfaces relevant memories and injects them into
  ``ProviderRequest.system_prompt``. The model receives them as
  context, no user-visible changes.
- :meth:`on_llm_response` runs after every LLM reply. It optionally
  triggers a fire-and-forget judgement task that decides whether the
  turn was worth auto-recording (when the model didn't proactively call
  ``record_memory``).

Both hooks must NEVER raise to the caller — that would break the user's
reply. We wrap each hook body in a top-level try/except.

Auto-record runs in a background task spawned via ``asyncio.create_task``
so the LLM judgement call doesn't add latency to the user-facing reply.
"""

from __future__ import annotations

import asyncio
import random
import re
from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import LLMResponse, ProviderRequest

from ..core.decay_engine import calculate_score
from ..core.prompts import MEMORY_PERSONA_PROMPT
from ..core.search_service import SearchHit
from ..core.surface_strategy import estimate_tokens
from .commands import (
    _event_is_group_chat,
    _extract_pairs,
    decorate_user_msg_with_speaker,
    format_digest_pairs,
)

if TYPE_CHECKING:
    from ..main import MemoryPlugin


# Use AstrBot's logger so messages route through the framework's log
# pipeline (visible in the dashboard log panel and standard stdout).
# A stdlib ``logging.getLogger("astrbot_plugin_ob_memory.hooks")``
# wouldn't reach the dashboard because AstrBot's handlers are bound to
# the "astrbot" logger only.


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
MEMORY_BLOCK_HEADER: str = "=== 长期记忆 ==="
MEMORY_BLOCK_FOOTER: str = "=== 记忆结束 ==="
"""Wrap injected memories in unmistakable headers so any debugging
session can trace where the prompt content came from."""

DEFAULT_AUTO_RECORD_MODE: str = "every_n_turns"
"""Auto-record dispatch mode. One of:

- ``per_turn``: legacy mode. After every chat turn the heuristic +
  LLM judge decide whether to record this single turn.
- ``every_n_turns``: counts turns per session and triggers a
  ``hold_diary`` summarisation of the last N turns once the counter
  reaches the threshold. Mirrors how ``/memory summarize N`` works,
  just on a timer.
- ``disabled``: never run the fallback. The model's own
  ``record_memory`` / ``record_feel`` / ``record_diary`` tool calls
  are the only path.
"""
DEFAULT_AUTO_RECORD_EVERY_N_TURNS: int = 20
"""How many user/assistant turns to accumulate before triggering an
auto-summary when ``auto_record_mode == 'every_n_turns'``."""

DEFAULT_AUTO_RECORD_MIN_CHARS: int = 60
DEFAULT_AUTO_RECORD_SKIP_PATTERNS: tuple[str, ...] = (
    r"^/",  # slash commands
    r"^[!！?？.。哈嗯啊嘿嘻噢哦呃唔～~…]+$",  # interjections only
    r"天气|weather",  # weather lookups
    r"^好的?$|^行$|^可以$|^ok$|^OK$|^好嘞$|^嗯嗯$|^是的$|^对$|^对啊$|^是$",
    r"^晚安$|^早安$|^午安$|^早$|^晚啦?$|^bye$|^88$|^拜拜$|^再见$",
    r"^你好$|^哈喽$|^hi$|^hello$|^在吗$|^在不在$",
    r"^谢谢$|^谢啦$|^thx$|^thanks$|^thank you$",
    r"^笑死$|^笑死我了?$|^草$|^哈哈+$|^呵呵+$|^嘿嘿+$|^嘻嘻+$",
    r"^\.{3,}$|^…+$",  # ellipses only
    r"^[\u4e00-\u9fff]$",  # a lone CJK character
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _session_id(event: AstrMessageEvent) -> str:
    """Resolve session id from event in origin mode (test fallback only).

    Production handlers prefer :func:`_resolve_session_via_plugin` so
    the user-configured ``scope_mode`` applies. This origin-only form
    remains so existing test stubs keep working.
    """
    sid = getattr(event, "unified_msg_origin", None)
    if not sid:
        sid = getattr(event, "session_id", None)
    return str(sid or "unknown")


async def _resolve_session_via_plugin(plugin, event: AstrMessageEvent) -> str:
    """Use the plugin's :class:`SessionResolver` if bound, else fallback.

    Tests bind hook methods onto stub objects without a resolver — they
    fall through to ``_session_id`` (origin) which is what the existing
    test fixtures expect.
    """
    resolver = getattr(plugin, "session_resolver", None)
    if resolver is None:
        return _session_id(event)
    try:
        return await resolver.resolve(event)
    except Exception:
        return _session_id(event)


async def _resolve_counter_key_via_plugin(plugin, event: AstrMessageEvent) -> str:
    """Resolve the ``every_n_turns`` counter key.

    Per-conversation (``conv:{cid}``) when AstrBot has a cid, otherwise
    falls back to ``origin:{umo}``. Decoupled from the memory
    session_id so users on ``scope_mode = user`` still get a fresh
    counter each time they open a new conversation window.

    Tests that don't bind a resolver fall through to the event's
    unified origin — same behaviour as before for legacy fixtures.
    """
    resolver = getattr(plugin, "session_resolver", None)
    if resolver is None or not hasattr(resolver, "resolve_counter_key"):
        return _session_id(event)
    try:
        return await resolver.resolve_counter_key(event)
    except Exception:
        return _session_id(event)


def _get_recent_pairs_buffer(plugin, session_id: str):
    """Return (and lazily create) the rolling ``(user, assistant)`` pair
    buffer for ``session_id``.

    Lives in memory on the plugin instance — lost across restarts,
    which is fine: the persistent counter still survives, and after a
    restart we just need to attend a few more turns to refill the
    window. Each session keeps the last ~64 pairs so even very large
    ``auto_record_every_n_turns`` settings have something to summarise.

    This is the fallback for adapters (notably group-chat adapters)
    where ``conversation_manager`` doesn't expose a usable cid and the
    standard history-fetch path returns empty.
    """
    from collections import deque

    buffers = getattr(plugin, "_recent_pairs", None)
    if buffers is None:
        buffers = {}
        try:
            plugin._recent_pairs = buffers
        except Exception:
            # Stub objects in tests may forbid attribute creation; just
            # build a throw-away buffer in that case.
            return deque(maxlen=64)
    buf = buffers.get(session_id)
    if buf is None:
        buf = deque(maxlen=64)
        buffers[session_id] = buf
    return buf


def _format_hit_for_injection(hit: SearchHit, *, snippet: str = "") -> str:
    """Render a single hit as one line for the injected memory block.

    ``snippet`` is the pre-dehydrated content (may be full or compressed).
    """
    bucket = hit.bucket
    name = bucket.name or bucket.id
    domain = "/".join(bucket.domain) if bucket.domain else "未分类"
    via_tag = "[语义关联]" if hit.via == "vector" else ""
    flag = "📌" if bucket.pinned else ("✅" if bucket.resolved else "")
    text = snippet or (bucket.content or "").strip().replace("\n", " ")
    return f"- {flag}{via_tag}[{domain}] {name}: {text} (id:{bucket.id})"


def _format_surfaced_for_injection(bucket, *, snippet: str = "") -> str:
    """Render a surfaced bucket; same shape as a hit but without score."""
    name = bucket.name or bucket.id
    domain = "/".join(bucket.domain) if bucket.domain else "未分类"
    flag = "📌" if bucket.pinned else ""
    text = snippet or (bucket.content or "").strip().replace("\n", " ")
    return f"- {flag}[{domain}] {name}: {text} (id:{bucket.id})"


def _injection_block(
    surfaced: list,
    hits: list[SearchHit],
    *,
    snippets: dict[str, str] | None = None,
) -> str:
    """Build the full injection block. Empty if both sources are empty.

    ``snippets`` maps bucket_id → dehydrated text. When provided, the
    pre-compressed text is used instead of raw content.
    """
    if not surfaced and not hits:
        return ""

    snips = snippets or {}
    parts: list[str] = [MEMORY_BLOCK_HEADER]
    if surfaced:
        parts.append("【最近浮现】")
        for bucket in surfaced:
            parts.append(
                _format_surfaced_for_injection(bucket, snippet=snips.get(bucket.id, ""))
            )
    if hits:
        parts.append("【相关回忆】")
        for hit in hits:
            parts.append(
                _format_hit_for_injection(hit, snippet=snips.get(hit.bucket.id, ""))
            )
    parts.append(MEMORY_BLOCK_FOOTER)
    return "\n".join(parts)


def _trim_to_budget(
    surfaced: list,
    hits: list[SearchHit],
    budget: int,
) -> tuple[list, list[SearchHit]]:
    """Drop lowest-priority items until the rendered block fits the budget.

    Priority order (highest to lowest survival):
    surfaced[0..n-1] (already in priority order from SurfaceStrategy)
    > hits[0] > hits[1] > ...

    We always keep at least one item across both lists if either is
    non-empty, otherwise the hook would produce nothing useful.
    """

    def render_size(s: list, h: list[SearchHit]) -> int:
        return estimate_tokens(_injection_block(s, h))

    surfaced = list(surfaced)
    hits = list(hits)

    if budget <= 0:
        return surfaced, hits

    # Drop lowest-ranked hits first.
    while hits and render_size(surfaced, hits) > budget:
        hits.pop()

    # If hits is now empty and surfaced still overflows, drop surfaced
    # from the bottom.
    while len(surfaced) > 1 and render_size(surfaced, hits) > budget:
        surfaced.pop()

    return surfaced, hits


def _compile_skip_patterns(raw: list[str] | tuple[str, ...] | None) -> list[re.Pattern]:
    if not raw:
        return []
    out: list[re.Pattern] = []
    for pat in raw:
        try:
            out.append(re.compile(pat))
        except re.error as e:
            logger.warning("invalid auto_record skip pattern %r: %s", pat, e)
    return out


def _heuristic_should_auto_record(
    user_msg: str,
    *,
    min_chars: int,
    skip_patterns: list[re.Pattern],
) -> bool:
    """Cheap pre-filter that runs before the LLM judgement call."""
    text = (user_msg or "").strip()
    if len(text) < min_chars:
        return False
    for pat in skip_patterns:
        if pat.search(text):
            return False
    return True


def _user_msg_from_request(req: ProviderRequest) -> str:
    """Extract the user-facing prompt for indexing / auto-record purposes."""
    if req.prompt:
        return str(req.prompt).strip()
    # Fall back to the latest user message in contexts.
    if req.contexts:
        for entry in reversed(req.contexts):
            if entry.get("role") == "user":
                content = entry.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    # OpenAI multi-part: pick text parts only.
                    parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                    return " ".join(p for p in parts if p).strip()
    return ""


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------
class MemoryHooksMixin:
    """``@on_llm_request`` and ``@on_llm_response`` hooks for ``MemoryPlugin``.

    Designed as a mixin so ``MemoryPlugin`` can opt-in by inheritance:

    ```python
    class MemoryPlugin(MemoryToolsMixin, MemoryHooksMixin, Star):
        ...
    ```

    Attributes used (set by ``MemoryPlugin`` during initialise):
    - ``manager``, ``writer``, ``search``, ``surface``, ``tagger``
    - ``config`` (plugin-level dict)
    """

    # Forward declarations for type checkers
    manager: object
    writer: object
    search: object
    surface: object
    tagger: object
    config: dict

    # ------------------------------------------------------------------
    # Config readers (re-evaluated per call so hot updates apply)
    # ------------------------------------------------------------------
    def _flat_cfg(self: MemoryPlugin) -> dict:
        """Return the plugin's flat config dict (all keys live at top level)."""
        return self.config if isinstance(self.config, dict) else {}

    def _is_session_disabled(self: MemoryPlugin, session_id: str) -> bool:
        # ``disabled_sessions`` is exposed in the advanced config section.
        # Listed sessions skip both memory injection and auto-record;
        # writes via ``/memory`` and LLM tools still work.
        disabled = self._flat_cfg().get("disabled_sessions") or []
        return session_id in (disabled if isinstance(disabled, list) else [])

    # ------------------------------------------------------------------
    # on_llm_request — inject memories into system_prompt
    # ------------------------------------------------------------------
    async def memory_on_llm_request(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """Augment the next LLM call with relevant memory context."""
        try:
            await self._inject_memories(event, req)
        except Exception as e:
            # Property 7: hook must never raise to the caller.
            logger.warning("memory injection skipped (caught error): %s", e)

    async def _inject_memories(
        self: MemoryPlugin,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """Core injection flow — runs inside the outer try/except."""
        if self.search is None or self.surface is None or self.manager is None:
            return  # plugin not fully initialised yet

        session_id = await _resolve_session_via_plugin(self, event)
        if self._is_session_disabled(session_id):
            return

        cfg = self._flat_cfg()

        # ----- Inject memory-behaviour persona prompt -----
        # Done before search/surface so even when no memories match this
        # turn, the model still gets the "you have a memory system"
        # context. Idempotent: skipped if the snippet is already present
        # (e.g. user pasted it into their persona by hand).
        if bool(cfg.get("inject_memory_persona", True)):
            self._inject_memory_persona(req)

        try:
            max_search = int(cfg.get("max_search_results", 3))
            max_surface = int(cfg.get("max_surface_results", 2))
            budget = int(cfg.get("injection_token_budget", 1500))
            random_drift_enabled = bool(cfg.get("random_drift_enabled", True))
            surface_recent = int(cfg.get("surface_recent_count", 1))
        except (TypeError, ValueError):
            max_search = 3
            max_surface = 2
            budget = 1500
            random_drift_enabled = True
            surface_recent = 1
        if surface_recent < 0:
            surface_recent = 0

        user_query = _user_msg_from_request(req)

        # ----- Search channel -----
        hits: list[SearchHit] = []
        if user_query and max_search > 0:
            try:
                hits = await self.search.search(  # type: ignore[union-attr]
                    session_id,
                    user_query,
                    limit=max_search,
                )
            except Exception as e:
                logger.debug("search channel skipped: %s", e)
                hits = []

        # ----- Surface channel -----
        surfaced = []
        if max_surface > 0:
            try:
                surfaced = await self.surface.surface(  # type: ignore[union-attr]
                    session_id,
                    token_budget=budget,
                    max_results=max_surface,
                    recent_count=surface_recent,
                )
            except Exception as e:
                logger.debug("surface channel skipped: %s", e)
                surfaced = []

        # Drop surfaced items that already appear in hits (avoid duplication).
        if surfaced and hits:
            hit_ids = {h.bucket.id for h in hits}
            surfaced = [b for b in surfaced if b.id not in hit_ids]

        # ----- Random drift -----
        # Mimics Ombre Brain's "忽然想起" behaviour: when both channels
        # return very little, occasionally surface a low-weight old bucket
        # to simulate the way human memory free-associates. Off-by-default
        # is the safest default; users can disable via config.
        drifted = await self._maybe_random_drift(
            session_id, hits, surfaced, enabled=random_drift_enabled
        )
        if drifted:
            # Append drifted to surfaced; trim budget will discard if it
            # doesn't fit, so we never blow the token budget.
            existing_ids = {b.id for b in surfaced} | {h.bucket.id for h in hits}
            for b in drifted:
                if b.id not in existing_ids:
                    surfaced.append(b)

        if not surfaced and not hits:
            return  # nothing to inject — leave req unchanged

        # ----- Token-budget trim -----
        surfaced, hits = _trim_to_budget(surfaced, hits, budget=budget)

        # ----- Dehydrate long content -----
        snippets: dict[str, str] = {}
        dehydrator = getattr(self, "_dehydrator", None)
        if dehydrator is not None:
            for bucket in surfaced:
                try:
                    snippets[bucket.id] = await dehydrator.dehydrate(
                        bucket.content, session_id=session_id
                    )
                except Exception:
                    pass
            for hit in hits:
                try:
                    snippets[hit.bucket.id] = await dehydrator.dehydrate(
                        hit.bucket.content, session_id=session_id
                    )
                except Exception:
                    pass

        block = _injection_block(surfaced, hits, snippets=snippets)
        if not block:
            return

        # ----- Inject into system_prompt -----
        existing = (req.system_prompt or "").rstrip()
        req.system_prompt = (existing + "\n\n" + block).strip() if existing else block

        # ----- touch only buckets reached by deliberate retrieval -----
        # Matches the upstream Ombre Brain semantics ("surfacing should NOT
        # reset decay timer", server.py at the same author): passive
        # weight-based surfacing must NOT mutate state, otherwise every
        # high-weight memory enters a positive-feedback loop where it
        # keeps surfacing because of its weight and keeps gaining
        # ``activation_count`` because it keeps surfacing. Only buckets
        # that matched a search query (``hits``) reflect a deliberate
        # retrieval and earn the bump.
        try:
            for hit in hits:
                await self.manager.touch(session_id, hit.bucket.id)  # type: ignore[union-attr]
        except Exception as e:
            logger.debug("touch loop after injection swallowed: %s", e)

        logger.debug(
            "[memory] injected session=%s surfaced=%d hits=%d",
            session_id,
            len(surfaced),
            len(hits),
        )

    # ------------------------------------------------------------------
    # Random drift helper
    # ------------------------------------------------------------------
    async def _maybe_random_drift(
        self: MemoryPlugin,
        session_id: str,
        hits: list,
        surfaced: list,
        *,
        enabled: bool,
    ) -> list:
        """Occasionally surface a low-weight old bucket to mimic human "忽然想起".

        Returns a list of 1-3 buckets to inject, or empty list when no
        drift should fire. Trigger conditions match Ombre Brain:

        - Total channels return < 3 results
        - 40% probability roll passes
        - At least one bucket exists with Activation_Score < 2.0

        The drifted buckets are ranked: oldest first feels most "out of
        the blue". We pick uniformly random from the eligible pool.
        """
        if not enabled:
            return []
        total = len(hits) + len(surfaced)
        if total >= 3:
            return []
        if random.random() >= 0.4:
            return []

        try:
            buckets = await self.manager.list_by_session(session_id)  # type: ignore[union-attr]
        except Exception:
            return []

        # Build the eligible pool: dynamic, non-resolved, non-pinned buckets
        # whose Activation_Score is genuinely "low" (< 2.0). This cleanly
        # excludes pinned (999), feel (50), permanent (999), recently
        # touched buckets (high freshness bonus).
        existing_ids = {b.id for b in surfaced} | {h.bucket.id for h in hits}
        pool = []
        for b in buckets:
            if b.id in existing_ids:
                continue
            if b.bucket_type != "dynamic" or b.pinned or b.resolved:
                continue
            try:
                score = calculate_score(b)
            except Exception:
                continue
            if score < 2.0:
                pool.append(b)

        if not pool:
            return []

        # Pick 1-3 at random.
        n = min(random.randint(1, 3), len(pool))
        return random.sample(pool, n)

    # ------------------------------------------------------------------
    # on_llm_response — fire-and-forget auto-record judgement
    # ------------------------------------------------------------------
    async def memory_on_llm_response(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        response: LLMResponse,
    ) -> None:
        """Optionally schedule a background auto-record task."""
        try:
            await self._maybe_schedule_auto_record(event, response)
        except Exception as e:
            logger.warning("auto-record scheduling skipped: %s", e)

    async def _maybe_schedule_auto_record(
        self: MemoryPlugin,
        event: AstrMessageEvent,
        response: LLMResponse,
    ) -> None:
        if self.writer is None or self.tagger is None:
            return

        cfg = self._flat_cfg()
        if not bool(cfg.get("auto_record_enabled", True)):
            return

        mode = str(cfg.get("auto_record_mode", DEFAULT_AUTO_RECORD_MODE)).lower()
        if mode == "disabled":
            return
        if mode not in ("per_turn", "every_n_turns"):
            logger.debug("[memory] unknown auto_record_mode=%r, falling back to %s", mode, DEFAULT_AUTO_RECORD_MODE)
            mode = DEFAULT_AUTO_RECORD_MODE

        session_id = await _resolve_session_via_plugin(self, event)
        if self._is_session_disabled(session_id):
            return

        # Counter key is intentionally distinct from ``session_id`` so a
        # ``scope_mode = user`` setup (memory shared across windows)
        # still gets a per-window summary cadence. See
        # ``SessionResolver.resolve_counter_key`` for the policy.
        counter_key = await _resolve_counter_key_via_plugin(self, event)

        # Skip the auto-record path if the model already recorded memory
        # via its own tool calls this turn. In ``every_n_turns`` mode we
        # simply don't count this turn toward the threshold (so we don't
        # re-summarise content the model already captured), but we keep
        # the existing progress so the user-visible counter doesn't reset
        # to 0 just because the model happened to record one turn.
        tools_called = list(getattr(response, "tools_call_name", []) or [])
        if any(
            tool_name in tools_called
            for tool_name in ("record_memory", "record_feel", "record_diary")
        ):
            if mode == "every_n_turns":
                current = await self._get_persisted_counter(counter_key)
                try:
                    n_threshold = int(
                        cfg.get(
                            "auto_record_every_n_turns",
                            DEFAULT_AUTO_RECORD_EVERY_N_TURNS,
                        )
                    )
                except (TypeError, ValueError):
                    n_threshold = DEFAULT_AUTO_RECORD_EVERY_N_TURNS
                logger.info(
                    "[memory] every_n_turns counter counter_key=%s session=%s %d/%d (turn not counted: model used %s)",
                    counter_key, session_id, current, n_threshold, tools_called,
                )
            else:
                logger.debug(
                    "[memory] auto-record skipped: tools already called %s", tools_called
                )
            return

        # Need a non-empty user message and assistant reply.
        user_msg = (
            getattr(event, "message_str", None)
            or getattr(event, "raw_message_str", None)
            or ""
        )
        user_msg = str(user_msg or "").strip()
        assistant_msg = (response.completion_text or "").strip()
        if not user_msg or not assistant_msg:
            logger.debug(
                "[memory] auto-record skipped: empty user_msg=%s or assistant_msg=%s",
                bool(user_msg),
                bool(assistant_msg),
            )
            return

        # Group-chat speaker enrichment: tag the user message with who
        # said it so a multi-speaker history doesn't get cross-attributed
        # by the digest LLM. No-op for private chats and for events
        # missing speaker info, so the wire format is unchanged in those
        # cases. See ``decorate_user_msg_with_speaker`` for the rationale.
        speaker_name: str | None = None
        if _event_is_group_chat(event):
            try:
                getter = getattr(event, "get_sender_name", None)
                raw = getter() if callable(getter) else None
            except Exception:
                raw = None
            if raw is None:
                raw = getattr(event, "sender_name", None)
            speaker_name = (str(raw).strip() if raw else None) or None
        user_msg = decorate_user_msg_with_speaker(user_msg, speaker_name)

        # Record this turn into the rolling buffer for fallback summarisation
        # when ``conversation_manager`` can't supply history (e.g. some
        # group-chat adapters). Keyed by the *memory* session_id so the
        # buffer follows memory scope — the summary content goes into the
        # same pool as everything else.
        try:
            _get_recent_pairs_buffer(self, session_id).append(
                (user_msg, assistant_msg)
            )
        except Exception as e:
            logger.debug("[memory] recent-pairs buffer append failed: %s", e)

        if mode == "every_n_turns":
            try:
                n = int(
                    cfg.get(
                        "auto_record_every_n_turns",
                        DEFAULT_AUTO_RECORD_EVERY_N_TURNS,
                    )
                )
            except (TypeError, ValueError):
                n = DEFAULT_AUTO_RECORD_EVERY_N_TURNS
            if n <= 0:
                return

            counter = await self._bump_persisted_counter(counter_key)
            if counter < n:
                logger.info(
                    "[memory] every_n_turns counter counter_key=%s session=%s %d/%d",
                    counter_key, session_id, counter, n,
                )
                return

            # Threshold reached — reset and trigger summary in background.
            await self._reset_persisted_counter(counter_key)
            logger.info(
                "[memory] every_n_turns threshold reached counter_key=%s session=%s %d/%d — triggering summary",
                counter_key, session_id, counter, n,
            )
            asyncio.create_task(self._auto_summary_task(event, session_id, n))
            return

        # ----- per_turn fallback (legacy behaviour) -----------------------
        try:
            min_chars = int(
                cfg.get("auto_record_min_chars", DEFAULT_AUTO_RECORD_MIN_CHARS)
            )
        except (TypeError, ValueError):
            min_chars = DEFAULT_AUTO_RECORD_MIN_CHARS

        skip_raw = cfg.get("auto_record_skip_patterns")
        if skip_raw is None:
            skip_raw = list(DEFAULT_AUTO_RECORD_SKIP_PATTERNS)
        skip_patterns = _compile_skip_patterns(skip_raw)

        if not _heuristic_should_auto_record(
            user_msg,
            min_chars=min_chars,
            skip_patterns=skip_patterns,
        ):
            return

        # Spawn the judgement + record flow in the background — must not
        # block the user-facing reply.
        is_group = _event_is_group_chat(event)
        asyncio.create_task(self._auto_record_task(
            session_id, user_msg, assistant_msg, group_context=is_group,
        ))

    def _inject_memory_persona(self: MemoryPlugin, req: ProviderRequest) -> None:
        """Append the memory-behaviour persona snippet to ``system_prompt``.

        The snippet itself is editable via the ``memory_persona_text``
        config field — leave it empty to use the plugin's built-in
        default (``MEMORY_PERSONA_PROMPT``), or paste a custom string in
        to override.

        Idempotent: if the *current* persona text is already present in
        ``system_prompt`` (e.g. user also pasted it into their AstrBot
        persona), we no-op so the model doesn't see two copies.

        Position: appended after the user's existing system_prompt and
        before the memory block. That ordering matches the natural
        reading flow:
        1. user's character identity (their persona)
        2. memory behaviour rules (this snippet)
        3. recalled memories for this turn (the memory block, added
           later in ``_inject_memories``).
        """
        cfg = self._flat_cfg()
        custom = str(cfg.get("memory_persona_text") or "").strip()
        persona = custom if custom else MEMORY_PERSONA_PROMPT.strip()
        if not persona:
            return
        existing = (req.system_prompt or "").rstrip()
        if persona in existing:
            return
        req.system_prompt = (existing + "\n\n" + persona).strip() if existing else persona

    async def _get_persisted_counter(self: MemoryPlugin, session_id: str) -> int:
        """Read the ``every_n_turns`` counter from SQLite.

        Persisted in ``session_state`` (see schema v2) so progress is
        preserved across plugin / AstrBot restarts. Returns 0 on any
        manager-level error so a transient DB hiccup just behaves like
        "fresh session" rather than crashing the request hook.
        """
        if self.manager is None:
            return 0
        try:
            return await self.manager.get_auto_record_counter(session_id)
        except Exception as e:
            logger.debug("[memory] persisted counter read failed: %s", e)
            return 0

    async def _bump_persisted_counter(
        self: MemoryPlugin, session_id: str
    ) -> int:
        """Atomically increment the persisted counter and return the new value."""
        if self.manager is None:
            return 0
        try:
            return await self.manager.bump_auto_record_counter(session_id)
        except Exception as e:
            logger.debug("[memory] persisted counter bump failed: %s", e)
            return 0

    async def _reset_persisted_counter(
        self: MemoryPlugin, session_id: str
    ) -> None:
        """Reset the persisted counter to 0 (called after a summary fires)."""
        if self.manager is None:
            return
        try:
            await self.manager.set_auto_record_counter(session_id, 0)
        except Exception as e:
            logger.debug("[memory] persisted counter reset failed: %s", e)

    async def _auto_summary_task(
        self: MemoryPlugin,
        event: AstrMessageEvent,
        session_id: str,
        n_turns: int,
    ) -> None:
        """Background body for ``every_n_turns`` mode: summarise last N
        user/assistant rounds via the existing ``hold_diary`` path.

        Resolution order for the pairs to summarise:

        1. ``conversation_manager`` history — preferred, includes
           context the model itself never echoed back (system prompts,
           tool calls, etc, get stripped by :func:`_extract_pairs`).
           Works on private chats and most adapters where AstrBot
           tracks a cid.
        2. In-memory rolling buffer (:func:`_get_recent_pairs_buffer`)
           populated turn-by-turn in :meth:`_maybe_schedule_auto_record`.
           This is the **group-chat-friendly fallback**: even when
           ``conversation_manager`` returns no history for this
           session, we still have the recent ``(user, assistant)``
           pairs sitting in plugin memory.

        Reuses :func:`_extract_pairs` from the commands module so we
        tolerate the same heterogeneous history shapes the
        ``/memory summarize`` command supports.
        """
        if self.writer is None:
            return

        pairs: list[tuple[str, str]] = []
        history_source = "none"

        # Path 1: AstrBot conversation history
        try:
            history, debug_info = await self._get_conversation_history(event)  # type: ignore[attr-defined]
        except Exception as e:
            logger.info(
                "[memory] auto-summary history fetch failed for session=%s: %s",
                session_id, e,
            )
            history = []
            debug_info = f"fetch raised: {e}"

        if history:
            extracted = _extract_pairs(history)
            if extracted:
                pairs = extracted
                history_source = "conversation_manager"
            else:
                logger.info(
                    "[memory] auto-summary: conversation_manager returned %d "
                    "messages but no user/assistant pairs (debug=%s)",
                    len(history), debug_info,
                )
        else:
            logger.info(
                "[memory] auto-summary: conversation_manager empty (debug=%s) "
                "\u2014 trying in-memory buffer fallback",
                debug_info,
            )

        # Path 2: rolling buffer fallback (group-chat-friendly)
        if not pairs:
            try:
                buf = _get_recent_pairs_buffer(self, session_id)
                if buf:
                    pairs = list(buf)
                    history_source = "rolling_buffer"
            except Exception as e:
                logger.info(
                    "[memory] auto-summary buffer fallback failed: %s", e
                )

        if not pairs:
            logger.info(
                "[memory] auto-summary skipped session=%s: no pairs available "
                "from any source",
                session_id,
            )
            return

        # Take the most recent N pairs only.
        pairs = pairs[-n_turns:]
        text = format_digest_pairs(pairs).strip()
        if not text:
            return

        # Cap the text size so we don't blow the tagger's context window
        # if N happens to be very large or messages are huge.
        if len(text) > 8000:
            text = text[-8000:]

        logger.info(
            "[memory] auto-summary firing session=%s pairs=%d source=%s",
            session_id, len(pairs), history_source,
        )

        is_group = _event_is_group_chat(event)

        try:
            result = await self.writer.hold_diary(
                session_id, text, group_context=is_group,
            )
        except Exception as e:
            logger.warning(
                "[memory] auto-summary hold_diary raised for session=%s: %s",
                session_id, e,
            )
            return

        entry_ids = [h.bucket_id for h in result.entries]
        logger.info(
            "[memory] auto-summary session=%s n_turns=%d ids=%s created=%s merged=%s failed=%s",
            session_id,
            len(pairs),
            entry_ids,
            result.created,
            result.merged,
            result.failed,
        )

    async def _auto_record_task(
        self: MemoryPlugin,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        *,
        group_context: bool = False,
    ) -> None:
        """Background body — judges + records. All errors swallowed."""
        cfg = self._flat_cfg()
        use_judge = bool(cfg.get("auto_record_use_judge", True))

        if use_judge:
            try:
                should, reason = await self.tagger.judge_worth_recording(  # type: ignore[union-attr]
                    user_msg,
                    assistant_msg,
                    session_id=session_id,
                )
            except Exception as e:
                logger.debug("auto-record judge raised: %s", e)
                return
            if not should:
                logger.debug("auto-record skipped (%s)", reason or "judge=no")
                return
        else:
            # No-judge mode: heuristic-only auto-record. The fact that we
            # got this far means the heuristic already passed in
            # ``_maybe_schedule_auto_record``; we just record.
            reason = "no-judge mode"

        content = format_digest_pairs([(user_msg, assistant_msg)]).strip()
        try:
            result = await self.writer.hold_diary(  # type: ignore[union-attr]
                session_id, content, group_context=group_context,
            )
        except Exception as e:
            logger.debug("auto-record hold_diary raised: %s", e)
            return
        entry_ids = [h.bucket_id for h in result.entries]
        logger.info(
            "[memory] auto-recorded session=%s ids=%s created=%s merged=%s failed=%s reason=%r",
            session_id,
            entry_ids,
            result.created,
            result.merged,
            result.failed,
            reason,
        )
