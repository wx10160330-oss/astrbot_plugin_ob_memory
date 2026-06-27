"""LLM-callable memory tools (function-calling).

Four tools are exposed so the model can manage memory autonomously:

- ``record_memory`` — store an event memory; auto-tags + merges
- ``record_feel`` — store the model's first-person reflection on a memory
- ``recall_memory`` — explicit search; returns top hits as plain text
- ``forget_memory`` — mark resolved (default) or hard-delete

The functions are written as a mixin so they live on the ``MemoryPlugin``
class. AstrBot's ``@filter.llm_tool`` decorator parses each function's
Google-style docstring to build the JSON schema sent to the LLM. We
follow that format exactly.

Critical security/behaviour rules baked in:

- Every call extracts ``session_id`` from
  ``event.unified_msg_origin``. The model is never trusted with a
  session id parameter — if it could pass one, a malicious prompt could
  read or pollute another user's memory.
- Every call traps internal exceptions and returns a human-readable
  string instead of raising — the LLM context already shows the user a
  reply; an exception bubbling up could break that reply.
- Every call ``await``s ``ensure_ready()`` which short-circuits to a
  user-readable error if the plugin failed to initialise. This keeps
  the failure mode predictable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent

from ..core.search_service import SearchHit

if TYPE_CHECKING:
    from ..main import MemoryPlugin


logger = logging.getLogger("astrbot_plugin_ob_memory.tools")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _session_id(event: AstrMessageEvent) -> str:
    """Resolve the AstrBot ``unified_msg_origin`` for the current event.

    Falls back to a string representation if the attribute is absent;
    in practice every real event has it, but tests use stubs.

    NOTE: this helper produces the *origin-mode* session_id and is kept
    for backwards-compatible test paths. Production handlers should
    call :meth:`MemoryToolsMixin._resolve_session` (or the equivalent on
    other mixins) so the configured ``scope_mode`` is honoured.
    """
    sid = getattr(event, "unified_msg_origin", None)
    if not sid:
        sid = str(getattr(event, "session_id", "")) or "unknown"
    return str(sid)


async def _resolve_session_via_plugin(plugin, event: AstrMessageEvent) -> str:
    """Use the plugin's :class:`SessionResolver` if available, else fall back.

    Tests bind handler methods to a stand-in object that may not own a
    resolver — when that's the case we degrade to ``_session_id``,
    keeping the existing test surface working unchanged.
    """
    resolver = getattr(plugin, "session_resolver", None)
    if resolver is None:
        return _session_id(event)
    try:
        return await resolver.resolve(event)
    except Exception:  # paranoia — resolver already swallows everything
        return _session_id(event)


def _split_csv(raw: str | None) -> list[str]:
    """Parse a comma-separated string into a clean list of non-empty values."""
    if not raw:
        return []
    out: list[str] = []
    for item in raw.split(","):
        s = item.strip()
        if s:
            out.append(s)
    return out


def _format_hit(hit: SearchHit, *, idx: int) -> str:
    """Render one search hit as a compact bullet for LLM consumption."""
    bucket = hit.bucket
    name = bucket.name or bucket.id
    domain = "/".join(bucket.domain) if bucket.domain else "未分类"
    via_marker = "[语义关联]" if hit.via == "vector" else ""
    snippet = (bucket.content or "").strip().replace("\n", " ")
    if len(snippet) > 120:
        snippet = snippet[:120] + "…"
    flags: list[str] = []
    if bucket.pinned:
        flags.append("📌")
    if bucket.resolved:
        flags.append("✅已解决")
    flag_str = "".join(flags)
    return (
        f"{idx}. {flag_str}{via_marker}[{domain}] {name} "
        f"(score={hit.score:.1f} | id={bucket.id}) — {snippet}"
    )


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------
class MemoryToolsMixin:
    """LLM tools mixed into ``MemoryPlugin``.

    The class has no ``__init__`` of its own; it relies on attributes set
    by :class:`MemoryPlugin` (``writer``, ``search``, ``manager``,
    ``embedding``).
    """

    # Attributes (forward-declared for type checkers)
    writer: object
    search: object
    manager: object
    embedding: object

    def _ready(self: MemoryPlugin) -> tuple[bool, str]:
        """Return ``(ok, error_message)`` so tools can short-circuit cleanly."""
        if self.manager is None or self.writer is None:
            return False, "记忆系统尚未初始化，请稍后再试。"
        return True, ""

    # ------------------------------------------------------------------
    # record_memory
    # ------------------------------------------------------------------
    async def record_memory(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        content: str,
        importance: int = 5,
        tags: str = "",
        pinned: bool = False,
    ) -> str:
        """记住一件事。AstrBot 会把这条记忆和当前会话绑定，并在以后相关对话里再带出来。

        Args:
            content(string): 要记住的内容，越具体越好。包含人物、时间、感受、待办等具体信息。
            importance(number): 1-10 的整数，1 表示水话别记、10 表示核心准则永不忘。默认 5。
            tags(string): 逗号分隔的关键词，方便日后检索；可以留空让系统自动生成。
            pinned(boolean): 是否钉选为永久核心准则；钉选后永不衰减。默认 false。
        """
        ok, err = self._ready()
        if not ok:
            return err
        if not content or not content.strip():
            return "记忆内容为空，未保存。"

        try:
            session_id = await _resolve_session_via_plugin(self, event)
            tag_list = _split_csv(tags)
            try:
                imp_int = int(importance)
            except (TypeError, ValueError):
                imp_int = 5
            result = await self.writer.hold(  # type: ignore[union-attr]
                session_id,
                content.strip(),
                importance=imp_int,
                tags=tag_list,
                pinned=bool(pinned),
            )
        except Exception as e:
            logger.warning("record_memory failed: %s", e)
            return f"记忆保存失败：{e}"

        bucket = result.target_bucket
        verb = "更新" if result.was_merged else "新建"
        pin_mark = "📌" if bucket.pinned else ""
        domain_str = "/".join(bucket.domain) if bucket.domain else "未分类"
        return (
            f"{pin_mark}{verb}记忆：{bucket.name or bucket.id} "
            f"(主题:{domain_str} 重要度:{bucket.importance} id:{bucket.id})"
        )

    # ------------------------------------------------------------------
    # record_feel
    # ------------------------------------------------------------------
    async def record_feel(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        content: str,
        source_bucket_id: str = "",
        valence: float = -1.0,
    ) -> str:
        """记下你（模型）从某段记忆里带走的感受。这与事件本身是分开的。

        Args:
            content(string): 你想记的第一人称感受、领悟或未解的疑问。例如「我从中看到了她的成长」。
            source_bucket_id(string): 这段感受对应的源事件桶 id；提供后会把源事件标为已消化。可留空。
            valence(number): 你对这段感受的效价 0.0-1.0；0 极负、0.5 中性、1 极正。-1 表示不指定。
        """
        ok, err = self._ready()
        if not ok:
            return err
        if not content or not content.strip():
            return "感受内容为空，未保存。"

        try:
            session_id = await _resolve_session_via_plugin(self, event)
            try:
                v = float(valence)
            except (TypeError, ValueError):
                v = -1.0
            v_arg = v if 0.0 <= v <= 1.0 else None
            result = await self.writer.hold_feel(  # type: ignore[union-attr]
                session_id,
                content.strip(),
                source_bucket_id=source_bucket_id.strip() or None,
                valence=v_arg,
            )
        except Exception as e:
            logger.warning("record_feel failed: %s", e)
            return f"感受保存失败：{e}"

        marked = "（源记忆已标记为已消化）" if result.source_marked_digested else ""
        return f"🫧已记下感受 (id:{result.bucket_id}){marked}"

    # ------------------------------------------------------------------
    # recall_memory
    # ------------------------------------------------------------------
    async def recall_memory(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        query: str = "",
        domain: str = "",
        limit: int = 10,
        importance_min: int = 0,
    ) -> str:
        """主动从记忆里检索相关内容。返回 top 命中条目供你接续回应。

        Args:
            query(string): 关键词或自然语言查询；可关于人物、事件、感受。可留空。
            domain(string): 可选的主题域过滤，例如「求职」「内心」。
                特别地传 "feel" 会进入 feel 独立通道，按时间倒序返回所有「感受」记忆。
            limit(number): 最多返回几条，默认 10，上限 20。
            importance_min(number): 1-10。设为 ≥1 时进入「批量拉重要记忆」模式：
                跳过语义检索，按 importance 降序返回 importance≥此值的桶。默认 0 表示关闭。
        """
        ok, err = self._ready()
        if not ok:
            return err

        try:
            session_id = await _resolve_session_via_plugin(self, event)
            try:
                lim = max(1, min(20, int(limit)))
            except (TypeError, ValueError):
                lim = 10
            try:
                imp_min = int(importance_min)
            except (TypeError, ValueError):
                imp_min = 0

            # ---------- Mode 1: importance batch fetch ----------
            if imp_min >= 1:
                buckets = await self.manager.list_by_session(session_id)  # type: ignore[union-attr]
                filtered = [
                    b
                    for b in buckets
                    if b.importance >= imp_min and b.bucket_type != "feel"
                ]
                filtered.sort(key=lambda b: -b.importance)
                filtered = filtered[:lim]
                if not filtered:
                    return f"没有重要度 ≥{imp_min} 的记忆。"
                lines = [f"重要度 ≥{imp_min} 的记忆 ({len(filtered)} 条)："]
                for idx, b in enumerate(filtered, start=1):
                    snippet = (b.content or "").strip().replace("\n", " ")
                    if len(snippet) > 120:
                        snippet = snippet[:120] + "…"
                    flags = "📌" if b.pinned else ""
                    lines.append(
                        f"{idx}. {flags}[importance:{b.importance}] "
                        f"{b.name or b.id} (id={b.id}) — {snippet}"
                    )
                return "\n".join(lines)

            # ---------- Mode 2: feel-only channel ----------
            if domain.strip().lower() == "feel":
                buckets = await self.manager.list_by_session(  # type: ignore[union-attr]
                    session_id, bucket_types=("feel",)
                )
                # list_by_session orders by last_active_at desc; we re-sort
                # by created_at desc to honour the spec for the feel channel.
                buckets.sort(key=lambda b: -b.created_at)
                buckets = buckets[:lim]
                if not buckets:
                    return "你还没有留下过 feel。"
                lines = [f"你留下的 feel ({len(buckets)} 条)："]
                for idx, b in enumerate(buckets, start=1):
                    snippet = (b.content or "").strip().replace("\n", " ")
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "…"
                    lines.append(f"{idx}. 🫧 V{b.valence:.1f} (id={b.id}) — {snippet}")
                return "\n".join(lines)

            # ---------- Mode 3: keyword + vector search ----------
            if not query or not query.strip():
                return '未提供检索关键词；如需读取感受请传 domain="feel"，或传 importance_min 拉重要记忆。'

            domain_filter = [domain.strip()] if domain and domain.strip() else None
            hits = await self.search.search(  # type: ignore[union-attr]
                session_id,
                query.strip(),
                limit=lim,
                domain_filter=domain_filter,
            )
        except Exception as e:
            logger.warning("recall_memory failed: %s", e)
            return f"检索过程出错：{e}"

        if not hits:
            return f"在记忆里没有找到「{query}」相关的条目。"

        # touch only the buckets we actually return — recall is a
        # deliberate retrieval (unlike passive surfacing which must not
        # mutate state).
        try:
            for hit in hits:
                await self.manager.touch(session_id, hit.bucket.id)  # type: ignore[union-attr]
        except Exception as e:
            logger.debug("recall touch loop swallowed: %s", e)

        lines = [f"检索到 {len(hits)} 条相关记忆："]
        for idx, hit in enumerate(hits, start=1):
            lines.append(_format_hit(hit, idx=idx))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # forget_memory
    # ------------------------------------------------------------------
    async def forget_memory(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        bucket_id: str,
        mode: str = "resolve",
    ) -> str:
        """让一段记忆退场。默认是「沉底」（仍可被关键词唤醒），传 mode=delete 才彻底删除。

        Args:
            bucket_id(string): 要操作的记忆桶 id；可以从 recall_memory 返回里拿到。
            mode(string): 处理方式：resolve 表示标记已解决（推荐），delete 表示永久删除（不可恢复）。
        """
        ok, err = self._ready()
        if not ok:
            return err
        if not bucket_id or not bucket_id.strip():
            return "未提供 bucket_id。"
        bid = bucket_id.strip()
        m = (mode or "resolve").strip().lower()

        try:
            session_id = await _resolve_session_via_plugin(self, event)
            if m == "delete":
                deleted = await self.manager.delete(session_id, bid)  # type: ignore[union-attr]
                return f"已永久删除记忆 {bid}。" if deleted else f"未找到记忆 {bid}。"
            # default: resolve (sink)
            updated = await self.manager.update(  # type: ignore[union-attr]
                session_id, bid, resolved=True
            )
            if updated is None:
                return f"未找到记忆 {bid}。"
            return f"已沉底记忆 {bid}（仍可被关键词唤醒）。"
        except Exception as e:
            logger.warning("forget_memory failed: %s", e)
            return f"操作失败：{e}"

    # ------------------------------------------------------------------
    # record_diary
    # ------------------------------------------------------------------
    async def record_diary(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        content: str,
    ) -> str:
        """把一大段日记/长文本拆分为多条独立记忆。适合用户一次倾诉很多内容时。

        系统会自动识别其中的事件、感受、决定、未完结的事，分别作为独立记忆存储。
        每条独立走一遍合并检测，相似话题会自动合并到已有桶。

        Args:
            content(string): 一段较长的内容；典型场景是用户当日的多件事汇总，或一段反思。
        """
        ok, err = self._ready()
        if not ok:
            return err
        if not content or not content.strip():
            return "日记内容为空，未保存。"

        try:
            session_id = await _resolve_session_via_plugin(self, event)
            result = await self.writer.hold_diary(session_id, content.strip())  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("record_diary failed: %s", e)
            return f"日记整理失败：{e}"

        if not result.entries:
            return "日记整理失败：没有产生任何记忆。"

        lines = [
            f"📒 日记已整理为 {len(result.entries)} 条记忆"
            + (f"（{result.failed} 条失败）" if result.failed else "")
            + f"，新建 {result.created} 条 / 合并 {result.merged} 条："
        ]
        for h in result.entries:
            b = h.target_bucket
            verb = "📎合并" if h.was_merged else "📝新建"
            domain_str = "/".join(b.domain) if b.domain else "未分类"
            lines.append(f"{verb} {b.name or b.id} ({domain_str}, id={b.id})")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # reflect_memory  (≈ Ombre Brain dream)
    # ------------------------------------------------------------------
    async def reflect_memory(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        limit: int = 10,
    ) -> str:
        """自省/做梦：读取最近几条记忆，引导你用第一人称想想哪些事还有重量、哪些可以放下。

        系统会返回最近创建的记忆 + 自省引导词；你看完后可以：
        - 觉得可以放下的：调用 forget_memory(bucket_id, mode='resolve') 让它沉底
        - 有沉淀的：调用 record_feel(content="...你的感受...", source_bucket_id="...") 写下感受
        - 没有沉淀就不写，不强迫产出

        典型用法：每次新对话开头调一次，把过去几天的事消化一下，然后开始正常对话。

        Args:
            limit(number): 最多读取几条最近记忆，默认 10，上限 20。
        """
        ok, err = self._ready()
        if not ok:
            return err

        try:
            session_id = await _resolve_session_via_plugin(self, event)
            try:
                lim = max(1, min(20, int(limit)))
            except (TypeError, ValueError):
                lim = 10

            buckets = await self.manager.list_by_session(  # type: ignore[union-attr]
                session_id
            )
        except Exception as e:
            logger.warning("reflect_memory failed: %s", e)
            return f"读取记忆失败：{e}"

        # Filter: only fresh dynamic non-resolved buckets that aren't pinned/feel
        # — these are the things that might still have weight.
        candidates = [
            b
            for b in buckets
            if b.bucket_type == "dynamic" and not b.pinned and not b.resolved
        ]
        if not candidates:
            return '没有需要消化的新记忆。如果想读以前留下的感受，可以调 recall_memory(domain="feel")。'

        # Sort by created_at desc, take top N (these are recently stored memories)
        candidates.sort(key=lambda b: -b.created_at)
        recent = candidates[:lim]

        lines = [
            "=== 自省时刻 ===",
            "以下是你最近的记忆。用第一人称想：",
            "- 这些事里有什么在你这里留下了重量？",
            "- 有什么还没想清楚？",
            "- 有什么可以放下了？",
            "",
            "想清楚之后：",
            "- 值得放下的 → forget_memory(bucket_id, mode='resolve')",
            "- 有沉淀的 → record_feel(content='...感受...', source_bucket_id='...')",
            "- 没有沉淀就不写，不强迫产出。valence 是你对这段记忆的感受，不是事件本身的情绪。",
            "",
            f"最近 {len(recent)} 条记忆：",
        ]
        for idx, b in enumerate(recent, start=1):
            snippet = (b.content or "").strip().replace("\n", " ")
            if len(snippet) > 150:
                snippet = snippet[:150] + "…"
            domain_str = "/".join(b.domain) if b.domain else "未分类"
            lines.append(
                f"{idx}. [{domain_str}] V{b.valence:.1f}/A{b.arousal:.1f} "
                f"{b.name or b.id} (id={b.id}) — {snippet}"
            )

        # Connection hint: find the most semantically similar pair via embeddings
        connection_hint = await self._reflect_connection_hint(session_id, recent)
        if connection_hint:
            lines.append("")
            lines.append(connection_hint)

        # Feel crystallisation hint: when ≥3 similar feels exist, suggest pinning
        crystal_hint = await self._reflect_crystal_hint(session_id)
        if crystal_hint:
            lines.append("")
            lines.append(crystal_hint)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers used only by reflect_memory (kept as methods so they can
    # access self.embedding / self.manager without re-resolving)
    # ------------------------------------------------------------------
    async def _reflect_connection_hint(
        self: MemoryPlugin, session_id: str, recent: list
    ) -> str:
        """Find the most-similar pair among ``recent`` via embeddings.

        Returns a one-line connection hint, or empty string when embedding
        is disabled or no pair clears the 0.5 similarity floor.
        """
        if self.embedding is None or not self.embedding.enabled:  # type: ignore[union-attr]
            return ""
        if len(recent) < 2:
            return ""
        try:
            from ..core.embedding_service import cosine_similarity

            embeddings = {}
            for b in recent:
                v = await self.embedding.get(b.id)  # type: ignore[union-attr]
                if v is not None:
                    embeddings[b.id] = (b, v)
            if len(embeddings) < 2:
                return ""

            best: tuple[float, str, str] | None = None
            ids = list(embeddings.keys())
            for i, id_a in enumerate(ids):
                _, vec_a = embeddings[id_a]
                for id_b in ids[i + 1 :]:
                    _, vec_b = embeddings[id_b]
                    sim = cosine_similarity(vec_a, vec_b)
                    if best is None or sim > best[0]:
                        best = (sim, id_a, id_b)

            if best is None or best[0] < 0.5:
                return ""

            sim, id_a, id_b = best
            ba = embeddings[id_a][0]
            bb = embeddings[id_b][0]
            return (
                f"💭 [{ba.name or ba.id}] 和 [{bb.name or bb.id}] 似乎有关联 "
                f"(相似度 {sim:.2f})——不替你下结论，你自己想想。"
            )
        except Exception as e:
            logger.debug("connection hint failed: %s", e)
            return ""

    async def _reflect_crystal_hint(self: MemoryPlugin, session_id: str) -> str:
        """Detect repeated feel themes worth crystallising into a pinned bucket.

        When ≥3 feel buckets cluster (similarity > 0.7 to ≥2 others), hint
        that the model could ``record_memory(pinned=True)`` to upgrade.
        """
        if self.embedding is None or not self.embedding.enabled:  # type: ignore[union-attr]
            return ""
        try:
            from ..core.embedding_service import cosine_similarity

            feels = await self.manager.list_by_session(  # type: ignore[union-attr]
                session_id, bucket_types=("feel",)
            )
            if len(feels) < 3:
                return ""
            embeddings = {}
            for f in feels:
                v = await self.embedding.get(f.id)  # type: ignore[union-attr]
                if v is not None:
                    embeddings[f.id] = (f, v)
            if len(embeddings) < 3:
                return ""

            for fid, (fb, fvec) in embeddings.items():
                if fb.pinned:
                    continue
                similar = 0
                for oid, (_, ovec) in embeddings.items():
                    if oid == fid:
                        continue
                    if cosine_similarity(fvec, ovec) > 0.7:
                        similar += 1
                if similar >= 2:
                    snippet = (fb.content or "").strip().replace("\n", " ")
                    if len(snippet) > 50:
                        snippet = snippet[:50] + "…"
                    return (
                        f"🔮 你已经写过 {similar + 1} 条相似的 feel "
                        f"（围绕「{snippet}」）。如果这已经成了确信而不只是感受，"
                        f"你可以用 record_memory(pinned=True) 升级它。不急，你自己决定。"
                    )
            return ""
        except Exception as e:
            logger.debug("crystal hint failed: %s", e)
            return ""
