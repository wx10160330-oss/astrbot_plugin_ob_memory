"""LLM-backed tagging, merging and judgement helpers.

The Tagger wraps three distinct LLM tasks needed by the memory pipeline:

1. **analyze**  — given a content snippet, derive ``domain / valence /
   arousal / tags / suggested_name / importance`` so it can be persisted
   with rich metadata.
2. **merge_content** — collapse two near-duplicate memory contents into a
   single passage that preserves both sets of facts.
3. **judge_worth_recording** — decide if a chat turn deserves long-term
   storage when the model didn't proactively call ``record_memory``.

Every method MUST degrade gracefully:
- No LLM provider available → return safe defaults, log a warning, never
  raise to the caller.
- Provider raises → same: log + fallback.
- Provider returns malformed JSON → safe parser falls back per field.

This is critical because the Tagger is called from the on_llm_request /
on_llm_response hooks. An exception there could break the user's reply.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from .prompts import ANALYZE_PROMPT, DIGEST_PROMPT, JUDGE_PROMPT, MERGE_PROMPT

if TYPE_CHECKING:
    from astrbot.core.provider.provider import Provider


logger = logging.getLogger("astrbot_plugin_ob_memory.tagger")


# ---------------------------------------------------------------------------
# Default analyze result. Used whenever the LLM call fails or output is
# unparseable. Numbers picked to be conservative neutral values.
# ---------------------------------------------------------------------------
DEFAULT_ANALYZE: dict[str, Any] = {
    "domain": ["未分类"],
    "valence": 0.5,
    "arousal": 0.3,
    "tags": [],
    "suggested_name": "",
    "importance": 5,
}


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(?P<body>.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _strip_code_fences(text: str) -> str:
    """Strip Markdown code fences if the model wrapped its JSON in them."""
    m = _JSON_FENCE_RE.match(text.strip())
    return m.group("body") if m else text


def _try_parse_json(text: str) -> Any | None:
    r"""Best-effort JSON parser that tolerates code fences and trailing prose.

    Order of attempts:
      1. ``json.loads`` on the raw input
      2. strip ``\`\`\`json``-style fences and retry
      3. extract the first balanced ``{...}`` substring and try that
      4. extract the first balanced ``[...]`` substring (for digest-style
         array outputs)

    Returns ``None`` if no valid JSON could be recovered.
    """
    if not text:
        return None
    candidates: list[str] = [text, _strip_code_fences(text)]
    # Heuristic: extract first {...} block by counting braces.
    opening = text.find("{")
    if opening != -1:
        depth = 0
        for i in range(opening, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[opening : i + 1])
                    break
    # Same trick for arrays.
    arr_opening = text.find("[")
    if arr_opening != -1:
        depth = 0
        for i in range(arr_opening, len(text)):
            ch = text[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidates.append(text[arr_opening : i + 1])
                    break
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _coerce_float(value: Any, default: float, lo: float, hi: float) -> float:
    """Robust float coercion with clamping and NaN guard."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _coerce_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _coerce_str_list(value: Any, max_items: int = 32) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:max_items]:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s:
            out.append(s)
    return out


def _normalize_analyze(raw: Any) -> dict[str, Any]:
    """Turn the LLM's free-form output into a clean, clamped dict."""
    if not isinstance(raw, dict):
        return dict(DEFAULT_ANALYZE)
    domain = _coerce_str_list(raw.get("domain"), max_items=4) or ["未分类"]
    tags = _coerce_str_list(raw.get("tags"), max_items=15)
    valence = _coerce_float(raw.get("valence"), 0.5, 0.0, 1.0)
    arousal = _coerce_float(raw.get("arousal"), 0.3, 0.0, 1.0)
    importance = _coerce_int(raw.get("importance"), 5, 1, 10)
    suggested_name = raw.get("suggested_name", "")
    if not isinstance(suggested_name, str):
        suggested_name = ""
    suggested_name = suggested_name.strip()[:40]
    return {
        "domain": domain,
        "valence": valence,
        "arousal": arousal,
        "tags": tags,
        "suggested_name": suggested_name,
        "importance": importance,
    }


def _normalize_judge(raw: Any) -> tuple[bool, str]:
    """Robust ``judge_worth_recording`` parser. Default is ``False`` —
    when in doubt, do not record (conservative default avoids spam)."""
    if not isinstance(raw, dict):
        return False, "解析失败"
    remember = bool(raw.get("remember", False))
    reason = raw.get("reason", "")
    if not isinstance(reason, str):
        reason = ""
    return remember, reason.strip()[:64]


def _normalize_digest(raw: Any) -> list[dict[str, Any]]:
    """Robust ``digest`` parser. Returns a list of clean entry dicts.

    Each entry has the same shape as :data:`DEFAULT_ANALYZE` plus a
    ``content`` field. Malformed entries are dropped silently. Returns
    an empty list when nothing usable was produced.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:10]:  # cap at 10 entries to bound work
        if not isinstance(item, dict):
            continue
        content = item.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        # Reuse the analyze normaliser for shared fields
        norm = _normalize_analyze(item)
        norm["content"] = content.strip()[:600]
        # Optional name override (analyze normaliser uses suggested_name)
        name = item.get("name", "")
        if isinstance(name, str) and name.strip():
            norm["suggested_name"] = name.strip()[:40]
        out.append(norm)
    return out


# ---------------------------------------------------------------------------
# Tagger
# ---------------------------------------------------------------------------
class Tagger:
    """LLM-backed analysis utilities.

    A Tagger is bound to an AstrBot ``Context`` so it can resolve the
    current text-chat provider via ``context.get_using_provider(umo=...)``
    on every call. This means the Tagger automatically follows the user's
    provider choices (per-session model selection, hot-swapping etc.).

    The constructor optionally accepts a fixed ``provider`` for tests; if
    set it bypasses the context lookup.
    """

    def __init__(
        self,
        context: Any,
        *,
        analyze_provider_id: str = "",
        fixed_provider: Provider | None = None,
    ):
        self.context = context
        self._analyze_provider_id = analyze_provider_id
        self._fixed_provider = fixed_provider

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------
    def _resolve_provider(self, session_id: str | None) -> Provider | None:
        """Pick the provider for this turn.

        If a fixed provider was injected (tests), use it. Otherwise:
        1. If analyze_provider_id is configured, try to get that specific provider
        2. Fall back to the active text-chat provider for the session
        """
        if self._fixed_provider is not None:
            return self._fixed_provider
        if self.context is None:
            return None
        try:
            # Try configured analyze provider first
            if self._analyze_provider_id:
                provider = self.context.get_provider_by_id(self._analyze_provider_id)
                if provider is not None:
                    return provider
                logger.warning(
                    "analyze provider %r not found, falling back to session provider",
                    self._analyze_provider_id,
                )
            # Fall back to session's active provider
            return self.context.get_using_provider(umo=session_id)
        except Exception as e:
            logger.warning("failed to resolve LLM provider: %s", e)
            return None

    async def _call(
        self,
        session_id: str | None,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Run a single chat completion. Returns ``""`` on failure."""
        provider = self._resolve_provider(session_id)
        if provider is None:
            return ""
        try:
            response = await provider.text_chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.warning("LLM text_chat failed: %s", e)
            return ""
        text = getattr(response, "completion_text", "") or ""
        return text.strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def analyze(
        self,
        content: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Auto-derive ``domain / valence / arousal / tags / suggested_name``.

        Falls back to :data:`DEFAULT_ANALYZE` on any failure (no provider,
        API error, malformed output). Always returns a valid dict, never
        raises.
        """
        if not content or not content.strip():
            return dict(DEFAULT_ANALYZE)

        raw_text = await self._call(
            session_id,
            system_prompt=ANALYZE_PROMPT,
            user_prompt=content,
        )
        if not raw_text:
            return dict(DEFAULT_ANALYZE)

        parsed = _try_parse_json(raw_text)
        return _normalize_analyze(parsed)

    async def merge_content(
        self,
        old_content: str,
        new_content: str,
        *,
        session_id: str | None = None,
    ) -> str:
        """Merge two related contents into one. On failure returns the
        concatenation, which is always a safe fallback (no information
        lost, only style suffers)."""
        if not new_content:
            return old_content
        if not old_content:
            return new_content

        prompt = f"OLD MEMORY:\n{old_content}\n\nNEW MEMORY:\n{new_content}\n"
        merged = await self._call(
            session_id,
            system_prompt=MERGE_PROMPT,
            user_prompt=prompt,
        )
        if not merged:
            return f"{old_content}\n\n{new_content}".strip()
        return merged

    async def judge_worth_recording(
        self,
        user_msg: str,
        assistant_msg: str,
        *,
        session_id: str | None = None,
    ) -> tuple[bool, str]:
        """Decide whether a (user, assistant) turn is worth storing.

        Returns ``(should_record, reason)``. The default on any failure
        is ``(False, ...)`` — favouring quiet over spam-recording when
        the LLM is offline.
        """
        if not user_msg.strip() or not assistant_msg.strip():
            return False, "空消息"

        prompt = f"USER: {user_msg.strip()}\nASSISTANT: {assistant_msg.strip()}\n"
        raw_text = await self._call(
            session_id,
            system_prompt=JUDGE_PROMPT,
            user_prompt=prompt,
        )
        if not raw_text:
            return False, "无 LLM"
        parsed = _try_parse_json(raw_text)
        return _normalize_judge(parsed)

    async def digest(
        self,
        content: str,
        *,
        session_id: str | None = None,
        digest_prompt_override: str | None = None,
    ) -> list[dict[str, Any]]:
        """Split a long passage into 2-6 standalone memory entries.

        Returns a list of dicts (each shaped like analyze output plus a
        ``content`` field). On any failure returns an empty list — the
        caller should fall back to treating the input as a single memory.

        ``digest_prompt_override`` allows the user to supply a custom
        system prompt via plugin config, overriding the built-in
        DIGEST_PROMPT.
        """
        if not content or not content.strip():
            return []

        prompt = digest_prompt_override.strip() if digest_prompt_override else ""
        system_prompt = prompt if prompt else DIGEST_PROMPT

        raw_text = await self._call(
            session_id,
            system_prompt=system_prompt,
            user_prompt=content,
        )
        if not raw_text:
            return []
        parsed = _try_parse_json(raw_text)
        return _normalize_digest(parsed)
