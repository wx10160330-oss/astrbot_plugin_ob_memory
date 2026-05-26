"""``/memory`` command group — user-facing CLI for the memory store.

Subcommands:

- ``/memory list [N]``       list top-N buckets in the current session by activation score
- ``/memory search <q>``     keyword + vector search; returns top 5 hits
- ``/memory pin <id>``       toggle pinned state on a bucket
- ``/memory forget <id>``    soft-resolve (still keyword-reachable)
- ``/memory delete <id> [confirm]``  permanent delete; two-step confirm
- ``/memory clear [confirm]`` admin-only nuke of the entire session; two-step confirm
- ``/memory stats``          counts per type + decay status

Two-step confirm pattern: a destructive subcommand without ``confirm``
records a *pending operation* in memory keyed by ``(session_id,
operation, target_id)``. The same subcommand re-issued with ``confirm``
within ``DESTRUCTIVE_TTL_SECONDS`` finds the pending op and executes it.
This is intentionally simple — no storage layer, just an in-process
dict — because confirmation is only meaningful within a single user's
short-term attention span.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrbot.api.event import AstrMessageEvent

from ..core.decay_engine import calculate_score
from ..core.models import MemoryBucket

if TYPE_CHECKING:
    from ..main import MemoryPlugin


logger = logging.getLogger("astrbot_plugin_ob_memory.commands")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DESTRUCTIVE_TTL_SECONDS: float = 300.0
"""Pending destructive ops expire after 5 minutes."""

DEFAULT_LIST_LIMIT: int = 10
LIST_HARD_CAP: int = 50
"""User can ask for more but we cap to avoid massive replies in groups."""

SEARCH_RESULT_LIMIT: int = 5
IMPORT_FILE_HARD_CAP_BYTES: int = 5 * 1024 * 1024
IMPORT_MAX_CONVERSATIONS: int = 20
IMPORT_MAX_ROUNDS_PER_CONVERSATION: int = 50

_ASTRBOT_SYSTEM_PROMPT_MARKERS: tuple[str, ...] = (
    "当前时间是:",
    "请你模拟系统设置的角色",
    "用户没有回复的次数是:",
    "请牢记本条消息并非用户所发",
)

_RAG_BLOCK_RE = re.compile(
    r"<RAG-Faiss-Memory>[\s\S]*?</RAG-Faiss-Memory>", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Pending-confirmation registry (process-local)
# ---------------------------------------------------------------------------
class _Pending:
    """Tiny TTL store keyed by (session_id, op_name).

    Each op stores its target id and timestamp. ``confirm()`` returns
    ``True`` only when the same key is re-confirmed before TTL expires.
    """

    def __init__(self):
        self._items: dict[tuple[str, str], tuple[str, float]] = {}

    def set(self, session_id: str, op: str, target: str) -> None:
        self._items[(session_id, op)] = (target, time.time())

    def pop_if_match(self, session_id: str, op: str, target: str) -> bool:
        key = (session_id, op)
        record = self._items.get(key)
        if record is None:
            return False
        stored_target, stamp = record
        if stored_target != target:
            return False
        if time.time() - stamp > DESTRUCTIVE_TTL_SECONDS:
            self._items.pop(key, None)
            return False
        self._items.pop(key, None)
        return True

    def clear(self) -> None:
        self._items.clear()


_pending = _Pending()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _session_id(event: AstrMessageEvent) -> str:
    """Resolve session id from event in origin mode (test fallback only).

    Production sub-commands prefer :func:`_resolve_session_via_plugin`
    so the user-configured ``scope_mode`` applies.
    """
    sid = getattr(event, "unified_msg_origin", None)
    if not sid:
        sid = getattr(event, "session_id", None)
    return str(sid or "unknown")


async def _resolve_session_via_plugin(plugin, event: AstrMessageEvent) -> str:
    """Use the plugin's :class:`SessionResolver` if bound, else fallback."""
    resolver = getattr(plugin, "session_resolver", None)
    if resolver is None:
        return _session_id(event)
    try:
        return await resolver.resolve(event)
    except Exception:
        return _session_id(event)


def _is_admin(event: AstrMessageEvent) -> bool:
    """Best-effort admin check.

    AstrBot exposes admin status via various attributes depending on the
    platform adapter. We look at the most common ones; missing attributes
    default to False (deny by default).
    """
    for attr in ("is_admin", "sender_is_admin"):
        if getattr(event, attr, False):
            return True
    role = (getattr(event, "sender_role", "") or "").lower()
    if role in ("admin", "owner", "supervisor"):
        return True
    return False


def _bucket_summary(bucket: MemoryBucket, *, score: float | None = None) -> str:
    """Render a bucket as one line for ``/memory list``."""
    name = bucket.name or bucket.id
    domain = "/".join(bucket.domain) if bucket.domain else "未分类"
    flags: list[str] = []
    if bucket.pinned:
        flags.append("📌")
    if bucket.bucket_type == "feel":
        flags.append("🫧")
    if bucket.bucket_type == "archived":
        flags.append("🗄️")
    if bucket.resolved:
        flags.append("✅")
    flag_str = "".join(flags)
    score_str = f" score={score:.2f}" if score is not None else ""
    snippet = (bucket.content or "").strip().replace("\n", " ")
    if len(snippet) > 80:
        snippet = snippet[:80] + "…"
    return (
        f"{flag_str}[{domain}] {name} (id:{bucket.id} imp:{bucket.importance}"
        f"{score_str}) — {snippet}"
    )


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------
class MemoryCommandsMixin:
    """``/memory`` command group, mixed into ``MemoryPlugin``.

    Each subcommand is a ``yield event.plain_result(...)`` async generator;
    AstrBot streams the result back to the user. Errors are caught and
    rendered as user-readable text, never re-raised.
    """

    # Forward declarations
    manager: object
    search: object
    decay: object

    # ------------------------------------------------------------------
    # /memory list [N]
    # ------------------------------------------------------------------
    async def cmd_memory_list(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        limit: int = DEFAULT_LIST_LIMIT,
    ):
        """列出当前会话最近活跃的记忆桶。"""
        if self.manager is None or self.decay is None:
            yield event.plain_result("memory: 未初始化")
            return
        try:
            n = max(1, min(LIST_HARD_CAP, int(limit)))
        except (TypeError, ValueError):
            n = DEFAULT_LIST_LIMIT
        sid = await _resolve_session_via_plugin(self, event)
        try:
            buckets = await self.manager.list_by_session(sid)  # type: ignore[union-attr]
        except Exception as e:
            yield event.plain_result(f"memory list 失败：{e}")
            return
        if not buckets:
            yield event.plain_result("当前会话还没有任何记忆。")
            return

        scored: list[tuple[float, MemoryBucket]] = []
        for b in buckets:
            try:
                s = calculate_score(b)
            except Exception:
                s = 0.0
            scored.append((s, b))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:n]

        lines = [f"=== 记忆 ({len(scored)}/{len(buckets)}) ==="]
        for s, b in scored:
            lines.append(_bucket_summary(b, score=s))
        yield event.plain_result("\n".join(lines))

    # ------------------------------------------------------------------
    # /memory search <query>
    # ------------------------------------------------------------------
    async def cmd_memory_search(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        query: str = "",
    ):
        """关键词或语义搜索。/memory search 实习"""
        if self.search is None:
            yield event.plain_result("memory: 未初始化")
            return
        if not query or not query.strip():
            yield event.plain_result("用法：/memory search <关键词>")
            return
        sid = await _resolve_session_via_plugin(self, event)
        try:
            hits = await self.search.search(  # type: ignore[union-attr]
                sid,
                query.strip(),
                limit=SEARCH_RESULT_LIMIT,
            )
        except Exception as e:
            yield event.plain_result(f"搜索失败：{e}")
            return
        if not hits:
            yield event.plain_result(f"没有找到「{query}」相关的记忆。")
            return
        lines = [f"找到 {len(hits)} 条相关记忆："]
        for h in hits:
            lines.append(_bucket_summary(h.bucket, score=h.score))
        yield event.plain_result("\n".join(lines))

    # ------------------------------------------------------------------
    # /memory pin <id>
    # ------------------------------------------------------------------
    async def cmd_memory_pin(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        bucket_id: str = "",
    ):
        """切换钉选状态。钉选的桶不衰减不合并。"""
        if self.manager is None:
            yield event.plain_result("memory: 未初始化")
            return
        bid = (bucket_id or "").strip()
        if not bid:
            yield event.plain_result("用法：/memory pin <bucket_id>")
            return
        sid = await _resolve_session_via_plugin(self, event)
        try:
            current = await self.manager.get(sid, bid)  # type: ignore[union-attr]
            if current is None:
                yield event.plain_result(f"未找到记忆 {bid}。")
                return
            new_pinned = not current.pinned
            updated = await self.manager.update(  # type: ignore[union-attr]
                sid,
                bid,
                pinned=new_pinned,
            )
        except Exception as e:
            yield event.plain_result(f"操作失败：{e}")
            return
        if updated is None:
            yield event.plain_result(f"未找到记忆 {bid}。")
            return
        yield event.plain_result("📌已钉选" if updated.pinned else "已取消钉选")

    # ------------------------------------------------------------------
    # /memory forget <id>
    # ------------------------------------------------------------------
    async def cmd_memory_forget(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        bucket_id: str = "",
    ):
        """让一段记忆沉底（仍可被关键词唤醒）。"""
        if self.manager is None:
            yield event.plain_result("memory: 未初始化")
            return
        bid = (bucket_id or "").strip()
        if not bid:
            yield event.plain_result("用法：/memory forget <bucket_id>")
            return
        sid = await _resolve_session_via_plugin(self, event)
        try:
            updated = await self.manager.update(  # type: ignore[union-attr]
                sid,
                bid,
                resolved=True,
            )
        except Exception as e:
            yield event.plain_result(f"操作失败：{e}")
            return
        if updated is None:
            yield event.plain_result(f"未找到记忆 {bid}。")
            return
        yield event.plain_result(f"已沉底记忆 {bid}（仍可被关键词唤醒）。")

    # ------------------------------------------------------------------
    # /memory delete <id> [confirm]
    # ------------------------------------------------------------------
    async def cmd_memory_delete(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        bucket_id: str = "",
        confirm: str = "",
    ):
        """永久删除一段记忆（不可恢复，需二次确认）。"""
        if self.manager is None:
            yield event.plain_result("memory: 未初始化")
            return
        bid = (bucket_id or "").strip()
        if not bid:
            yield event.plain_result("用法：/memory delete <bucket_id>")
            return
        sid = await _resolve_session_via_plugin(self, event)

        # Check existence first so we don't ask for confirmation on a
        # nonexistent id.
        try:
            existing = await self.manager.get(sid, bid)  # type: ignore[union-attr]
        except Exception as e:
            yield event.plain_result(f"操作失败：{e}")
            return
        if existing is None:
            yield event.plain_result(f"未找到记忆 {bid}。")
            return

        if (confirm or "").strip().lower() != "confirm":
            _pending.set(sid, "delete", bid)
            yield event.plain_result(
                f"⚠️ 即将永久删除：{existing.name or bid}\n"
                f"这是不可恢复的操作。\n"
                f"请在 5 分钟内再次发送：/memory delete {bid} confirm"
            )
            return

        if not _pending.pop_if_match(sid, "delete", bid):
            yield event.plain_result(
                f"确认已过期或不匹配。请重新执行 /memory delete {bid}。"
            )
            return

        try:
            ok = await self.manager.delete(sid, bid)  # type: ignore[union-attr]
        except Exception as e:
            yield event.plain_result(f"删除失败：{e}")
            return
        if ok:
            yield event.plain_result(f"已永久删除记忆 {bid}。")
        else:
            yield event.plain_result(f"未找到记忆 {bid}。")

    # ------------------------------------------------------------------
    # /memory clear [confirm]
    # ------------------------------------------------------------------
    async def cmd_memory_clear(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        confirm: str = "",
    ):
        """清空当前会话的所有记忆（管理员，需二次确认）。"""
        if self.manager is None:
            yield event.plain_result("memory: 未初始化")
            return
        if not _is_admin(event):
            yield event.plain_result("此操作需要管理员权限。")
            return
        sid = await _resolve_session_via_plugin(self, event)
        try:
            buckets = await self.manager.list_by_session(  # type: ignore[union-attr]
                sid,
                include_archived=True,
            )
        except Exception as e:
            yield event.plain_result(f"操作失败：{e}")
            return
        if not buckets:
            yield event.plain_result("当前会话没有需要清除的记忆。")
            return

        if (confirm or "").strip().lower() != "confirm":
            _pending.set(sid, "clear", "*")
            yield event.plain_result(
                f"⚠️ 即将清空当前会话的全部 {len(buckets)} 条记忆。\n"
                f"这是不可恢复的操作。\n"
                f"请在 5 分钟内再次发送：/memory clear confirm"
            )
            return

        if not _pending.pop_if_match(sid, "clear", "*"):
            yield event.plain_result("确认已过期，请重新执行 /memory clear。")
            return

        deleted = 0
        for b in buckets:
            try:
                if await self.manager.delete(sid, b.id):  # type: ignore[union-attr]
                    deleted += 1
            except Exception as e:
                logger.warning("clear: delete %s failed: %s", b.id, e)
        yield event.plain_result(f"已清空 {deleted} 条记忆。")

    # ------------------------------------------------------------------
    # /memory stats
    # ------------------------------------------------------------------
    async def cmd_memory_stats(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
    ):
        """显示当前会话的记忆系统状态。"""
        if self.manager is None:
            yield event.plain_result("memory: 未初始化")
            return
        sid = await _resolve_session_via_plugin(self, event)
        try:
            counts = await self.manager.count_in_session(sid)  # type: ignore[union-attr]
        except Exception as e:
            yield event.plain_result(f"操作失败：{e}")
            return

        # Counter is keyed per-conversation, not per-memory-session, so
        # ``/memory stats`` reads the same key the auto-record hook
        # writes to. See ``SessionResolver.resolve_counter_key``.
        counter_key = sid
        resolver = getattr(self, "session_resolver", None)
        if resolver is not None and hasattr(resolver, "resolve_counter_key"):
            try:
                counter_key = await resolver.resolve_counter_key(event)
            except Exception:
                counter_key = sid

        decay_status = "未运行"
        if self.decay is not None:
            try:
                decay_status = "运行中" if self.decay.is_running else "已停止"  # type: ignore[union-attr]
            except Exception:
                decay_status = "未知"

        embedding_status = "未启用"
        if getattr(self, "embedding", None) is not None:
            try:
                embedding_status = (
                    "已启用" if self.embedding.enabled else "未启用"  # type: ignore[union-attr]
                )
            except Exception:
                embedding_status = "未知"

        total = sum(counts.values())

        # Auto-record mode + counter snapshot (every_n_turns progress).
        raw_cfg = getattr(self, "config", None)
        cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
        auto_record_enabled = bool(cfg.get("auto_record_enabled", True))
        mode = str(cfg.get("auto_record_mode", "every_n_turns")).lower()
        if not auto_record_enabled:
            auto_record_line = "兜底自动记录: 已关闭"
        elif mode == "disabled":
            auto_record_line = "兜底自动记录: disabled（只靠模型主动）"
        elif mode == "per_turn":
            auto_record_line = "兜底自动记录: per_turn（每轮判断）"
        elif mode == "every_n_turns":
            try:
                n_threshold = int(cfg.get("auto_record_every_n_turns", 20))
            except (TypeError, ValueError):
                n_threshold = 20
            try:
                current = await self.manager.get_auto_record_counter(counter_key)  # type: ignore[union-attr]
            except Exception:
                current = 0
            remaining = max(0, n_threshold - current)
            # The "(本对话)" suffix matters when ``scope_mode`` makes the
            # memory session_id broader than the counter key (e.g. the
            # user has all their windows pointing at a shared ``user:``
            # memory pool but each window has its own ``conv:`` cadence).
            scope_hint = "本对话" if counter_key != sid else "本会话"
            auto_record_line = (
                f"兜底自动记录: every_n_turns {current}/{n_threshold}"
                f"（距下次自动总结还差 {remaining} 轮 · {scope_hint}）"
            )
        else:
            auto_record_line = f"兜底自动记录: 未知模式 {mode!r}"

        yield event.plain_result(
            f"=== 记忆系统状态 ({sid}) ===\n"
            f"动态: {counts.get('dynamic', 0)}\n"
            f"钉选/永久: {counts.get('permanent', 0)}\n"
            f"感受 (feel): {counts.get('feel', 0)}\n"
            f"已归档: {counts.get('archived', 0)}\n"
            f"合计: {total}\n"
            f"衰减引擎: {decay_status}\n"
            f"向量检索: {embedding_status}\n"
            f"{auto_record_line}"
        )

    # ------------------------------------------------------------------
    # /memory help
    # ------------------------------------------------------------------
    async def cmd_memory_help(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
    ):
        """显示子指令列表。"""
        yield event.plain_result(
            "=== /memory 子指令 ===\n"
            "/memory list [N]              列出最近活跃的桶\n"
            "/memory search <关键词>        关键词+向量搜索\n"
            "/memory summarize [N]         总结最近 N 轮对话为记忆\n"
            "/memory import_astrbot <文件路径> [N]  导入 AstrBot JSONL 历史\n"
            "/memory pin <id>               切换钉选状态\n"
            "/memory forget <id>            沉底（仍可被唤醒）\n"
            "/memory delete <id> [confirm]  永久删除（二次确认）\n"
            "/memory clear [confirm]        清空当前会话（管理员，二次确认）\n"
            "/memory stats                  系统状态"
        )

    # ------------------------------------------------------------------
    # /memory import_astrbot <path> [max_pairs]
    # ------------------------------------------------------------------
    async def cmd_memory_import_astrbot(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        file_path: str = "",
        max_pairs: int = 30,
    ):
        """从 AstrBot 导出的 JSONL 历史中提取记忆。"""
        if self.writer is None or self.manager is None:
            yield event.plain_result("memory: 未初始化")
            return

        raw_path = (file_path or "").strip().strip('"')
        if not raw_path:
            yield event.plain_result(
                "用法：/memory import_astrbot <jsonl文件路径> [最多导入轮数]"
            )
            return

        path = Path(raw_path)
        if path.suffix.lower() != ".jsonl":
            yield event.plain_result("目前只支持 AstrBot 导出的 .jsonl 文件。")
            return
        if not path.exists() or not path.is_file():
            yield event.plain_result(f"文件不存在：{raw_path}")
            return
        if path.stat().st_size > IMPORT_FILE_HARD_CAP_BYTES:
            yield event.plain_result("文件过大，请先拆分到 5MB 以内再导入。")
            return

        try:
            limit = max(1, min(200, int(max_pairs)))
        except (TypeError, ValueError):
            limit = 30

        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            yield event.plain_result("文件不是 UTF-8 编码，暂不支持。")
            return
        except OSError as e:
            yield event.plain_result(f"读取文件失败：{e}")
            return

        pairs = _extract_pairs_from_astrbot_jsonl(raw_text)
        if not pairs:
            yield event.plain_result("没有从该 JSONL 中识别到可导入的用户/助手对话。")
            return

        if len(pairs) > limit:
            pairs = pairs[:limit]

        full_text = format_digest_pairs(pairs)
        if len(full_text) > 12000:
            full_text = full_text[:12000]

        sid = await _resolve_session_via_plugin(self, event)
        yield event.plain_result(
            f"📝 正在从 AstrBot 历史中导入 {len(pairs)} 轮对话，请稍候..."
        )

        try:
            result = await self.writer.hold_diary(sid, full_text)  # type: ignore[union-attr]
        except Exception as e:
            yield event.plain_result(f"导入失败：{e}")
            return

        if not result.entries:
            yield event.plain_result("导入完成，但没有产生值得记住的内容。")
            return

        lines = [
            f"✅ 从 AstrBot 历史中提取了 {len(result.entries)} 条记忆"
            + (f"（{result.failed} 条失败）" if result.failed else "")
            + f"，新建 {result.created} 条 / 合并 {result.merged} 条："
        ]
        for h in result.entries:
            b = h.target_bucket
            verb = "📎合并" if h.was_merged else "📝新建"
            domain_str = "/".join(b.domain) if b.domain else "未分类"
            lines.append(f"  {verb} {b.name or b.id} ({domain_str}, id={b.id})")
        yield event.plain_result("\n".join(lines))

    # ------------------------------------------------------------------
    # /memory summarize [N]
    # ------------------------------------------------------------------
    async def cmd_memory_summarize(  # type: ignore[override]
        self: MemoryPlugin,
        event: AstrMessageEvent,
        rounds: int = 0,
    ):
        """从当前对话上下文中提取记忆。N 为轮次数，0 表示全部。"""
        if self.writer is None or self.manager is None:
            yield event.plain_result("memory: 未初始化")
            return

        sid = await _resolve_session_via_plugin(self, event)

        # Determine how many rounds to summarize
        cfg = getattr(self, "config", {}) or {}
        default_rounds = int(cfg.get("summarize_default_rounds", 0))
        try:
            n = int(rounds)
        except (TypeError, ValueError):
            n = 0
        if n <= 0:
            n = default_rounds  # 0 means "all"

        # Fetch conversation history from AstrBot. We try conversation_manager
        # first (works for non-WebChat platforms and AstrBot < 4.0) and fall
        # back to PlatformMessageHistory (WebChat in v4+).
        try:
            history, debug = await self._get_conversation_history(event)
        except Exception as e:
            yield event.plain_result(f"获取对话历史失败：{e}")
            return

        if not history:
            yield event.plain_result(
                "当前对话没有历史记录，无法总结。\n"
                f"（调试信息：{debug}）"
            )
            return

        # Extract user-assistant pairs (one "round" = one user + one assistant)
        pairs = _extract_pairs(history)
        if not pairs:
            shape = _describe_history_shape(history)
            yield event.plain_result(
                "当前对话没有可总结的内容。\n"
                f"（拿到 {len(history)} 条历史但没能拼出 user/assistant 轮次；"
                f"{debug}；样本：{shape}）"
            )
            return

        # Limit to last N rounds if specified
        if n > 0 and len(pairs) > n:
            pairs = pairs[-n:]

        full_text = format_digest_pairs(pairs)

        # Truncate if extremely long (LLM context limit protection)
        max_chars = 8000
        if len(full_text) > max_chars:
            full_text = full_text[-max_chars:]

        yield event.plain_result(
            f"📝 正在总结最近 {len(pairs)} 轮对话为记忆，请稍候..."
        )

        # Use hold_diary to split and store
        try:
            result = await self.writer.hold_diary(sid, full_text)  # type: ignore[union-attr]
        except Exception as e:
            yield event.plain_result(f"总结失败：{e}")
            return

        if not result.entries:
            yield event.plain_result("总结完成，但没有产生值得记住的内容。")
            return

        lines = [
            f"✅ 从 {len(pairs)} 轮对话中提取了 {len(result.entries)} 条记忆"
            + (f"（{result.failed} 条失败）" if result.failed else "")
            + f"，新建 {result.created} 条 / 合并 {result.merged} 条："
        ]
        for h in result.entries:
            b = h.target_bucket
            verb = "📎合并" if h.was_merged else "📝新建"
            domain_str = "/".join(b.domain) if b.domain else "未分类"
            lines.append(f"  {verb} {b.name or b.id} ({domain_str}, id={b.id})")
        yield event.plain_result("\n".join(lines))

    async def _get_conversation_history(
        self: MemoryPlugin, event: AstrMessageEvent
    ) -> tuple[list[dict], str]:
        """Retrieve the current conversation's message history.

        Returns ``(history, debug_info)``. ``history`` is a list of
        OpenAI-format messages (possibly empty). ``debug_info`` is a short
        string describing which paths were tried and why they returned
        empty, surfaced to the user when summarize finds nothing to chew on
        so we can tell whether the problem is the conversation manager,
        the platform message history table, or just a genuinely empty
        conversation.
        """
        umo = getattr(event, "unified_msg_origin", None) or "<unknown>"
        notes: list[str] = []

        # Path 1: conversation_manager -> Conversation.history (works for
        # non-WebChat platforms and AstrBot < 4.0).
        conv_mgr = getattr(self.context, "conversation_manager", None)
        cid: str | None = None
        if conv_mgr is None:
            notes.append("no conversation_manager")
        else:
            try:
                cid = await conv_mgr.get_curr_conversation_id(umo)
            except Exception as e:
                notes.append(f"get_curr_conversation_id err: {e}")
                cid = None
            if not cid:
                notes.append(f"no cid for umo={umo}")
            else:
                try:
                    conversation = await conv_mgr.get_conversation(umo, cid)
                except Exception as e:
                    notes.append(f"get_conversation err: {e}")
                    conversation = None
                if not conversation:
                    notes.append(f"no conversation for cid={cid}")
                elif not conversation.history:
                    notes.append(f"conversation.history empty for cid={cid}")
                else:
                    try:
                        parsed = json.loads(conversation.history)
                    except (json.JSONDecodeError, TypeError) as e:
                        notes.append(f"history json parse err: {e}")
                        parsed = None
                    if isinstance(parsed, list) and parsed:
                        return parsed, "; ".join(notes) or "ok via conversation_manager"
                    elif isinstance(parsed, list):
                        notes.append("conversation history list empty")
                    else:
                        notes.append(
                            f"conversation history not a list (type={type(parsed).__name__})"
                        )

        # Path 2: PlatformMessageHistory (WebChat in AstrBot v4+; for other
        # platforms this table is generally empty).
        msg_history_mgr = getattr(self.context, "message_history_manager", None)
        if msg_history_mgr is None:
            notes.append("no message_history_manager")
        else:
            platform_id: str | None = None
            try:
                if hasattr(event, "get_platform_id"):
                    platform_id = event.get_platform_id()
            except Exception as e:
                notes.append(f"get_platform_id err: {e}")
                platform_id = None

            # WebChat stores PlatformMessageHistory under user_id=cid;
            # other platforms (if they use it at all) use the umo. We try
            # both so we don't have to special-case WebChat by name.
            user_id_candidates: list[str] = []
            if cid:
                user_id_candidates.append(cid)
            if umo and umo not in user_id_candidates:
                user_id_candidates.append(umo)

            platform_id_candidates = [platform_id] if platform_id else []
            if "webchat" not in platform_id_candidates:
                platform_id_candidates.append("webchat")

            for pid in platform_id_candidates:
                for uid in user_id_candidates:
                    try:
                        records = await msg_history_mgr.get(
                            platform_id=pid,
                            user_id=uid,
                            page=1,
                            page_size=500,
                        )
                    except Exception as e:
                        notes.append(f"msg_history.get({pid},{uid}) err: {e}")
                        continue
                    if not records:
                        notes.append(
                            f"msg_history.get({pid},{uid}) empty"
                        )
                        continue
                    converted = _convert_platform_history(records)
                    if converted:
                        return converted, (
                            f"ok via message_history({pid},{uid}) "
                            f"with {len(converted)} msgs"
                        )
                    notes.append(
                        f"msg_history({pid},{uid}) had {len(records)} "
                        f"records but none convertible"
                    )

        return [], "; ".join(notes) or "no history sources available"


def format_digest_pairs(pairs: list[tuple[str, str]]) -> str:
    """Format user-assistant pairs for digest input.

    Exported for use by llm_hooks.py auto-record flow.
    """
    text_parts: list[str] = []
    for user_msg, assistant_msg in pairs:
        text_parts.append(f"对方(用户)说: {user_msg}")
        text_parts.append(f"我(AI)回应: {assistant_msg}")
    return "\n".join(text_parts)


_ASSISTANT_SENDER_HINTS = {"bot", "assistant", "ai", "model", "system_bot"}


def _convert_platform_history(records: list[Any]) -> list[dict]:
    """Convert ``PlatformMessageHistory`` rows into OpenAI-format messages.

    AstrBot >= 4.0 stores WebChat history in the ``PlatformMessageHistory``
    table instead of ``Conversation.history``. Each row is one message
    (user OR bot), with the actual text living under various keys
    depending on the platform adapter.

    Returns a list of ``{"role": ..., "content": ...}`` dicts in
    chronological order (oldest first). Records that have no extractable
    text content are silently skipped.
    """
    out: list[dict] = []
    for rec in records:
        content = getattr(rec, "content", None)
        if content is None:
            continue
        text = ""
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, dict):
            for key in ("message", "text", "content", "plain_text"):
                v = content.get(key)
                if isinstance(v, str) and v.strip():
                    text = v.strip()
                    break
            if not text:
                parts = content.get("parts") or content.get("message_parts")
                if isinstance(parts, list):
                    chunks = []
                    for p in parts:
                        if isinstance(p, str) and p.strip():
                            chunks.append(p.strip())
                        elif isinstance(p, dict):
                            for k in ("text", "message", "content"):
                                vv = p.get(k)
                                if isinstance(vv, str) and vv.strip():
                                    chunks.append(vv.strip())
                                    break
                    if chunks:
                        text = "\n".join(chunks).strip()
        if not text:
            continue

        sender_name = (getattr(rec, "sender_name", None) or "").lower()
        sender_id = (getattr(rec, "sender_id", None) or "").lower()
        ctype = ""
        if isinstance(content, dict):
            raw_type = content.get("type") or content.get("role") or ""
            if isinstance(raw_type, str):
                ctype = raw_type.lower()

        role = "user"
        if (
            sender_name in _ASSISTANT_SENDER_HINTS
            or sender_id in _ASSISTANT_SENDER_HINTS
            or ctype in _ASSISTANT_SENDER_HINTS
        ):
            role = "assistant"

        out.append({"role": role, "content": text})
    return out


_USER_ROLE_HINTS = {"user", "human", "you"}
_ASSISTANT_ROLE_HINTS = {"assistant", "ai", "bot", "model", "system_bot"}
_SKIP_ROLE_HINTS = {"tool", "function", "system", "_checkpoint", "developer"}


def _msg_text(msg: dict) -> str:
    """Best-effort extract human-readable text out of a single history
    message regardless of which AstrBot / provider schema produced it."""
    if not isinstance(msg, dict):
        return ""

    content = msg.get("content")
    if content is None:
        # Some adapters store the text under 'message' / 'text'.
        for k in ("message", "text", "plain_text"):
            v = msg.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str) and p.strip():
                parts.append(p.strip())
                continue
            if isinstance(p, dict):
                # OpenAI multimodal: {"type": "text", "text": "..."}
                if (
                    p.get("type") == "text"
                    and isinstance(p.get("text"), str)
                    and p["text"].strip()
                ):
                    parts.append(p["text"].strip())
                    continue
                # Some schemas put the text directly under "content".
                for k in ("text", "message", "content", "plain_text"):
                    v = p.get(k)
                    if isinstance(v, str) and v.strip():
                        parts.append(v.strip())
                        break
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        for k in ("text", "message", "content", "plain_text"):
            v = content.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return ""


def _msg_role(msg: dict) -> str:
    """Classify a history message as 'user' / 'assistant' / 'skip' / 'unknown'."""
    if not isinstance(msg, dict):
        return "unknown"
    raw = (
        msg.get("role")
        or msg.get("type")
        or msg.get("sender_type")
        or msg.get("sender")
        or ""
    )
    raw = str(raw).strip().lower()
    if raw in _USER_ROLE_HINTS:
        return "user"
    if raw in _ASSISTANT_ROLE_HINTS:
        return "assistant"
    if raw in _SKIP_ROLE_HINTS:
        return "skip"
    return "unknown"


def _extract_pairs(history: list[dict]) -> list[tuple[str, str]]:
    """Extract (user_msg, assistant_msg) pairs from a history list.

    Tolerates several common shapes:
    - OpenAI: ``{"role": "user"|"assistant", "content": str | list[dict]}``
    - AstrBot WebChat legacy: ``{"type": "user"|"bot", "message": "..."}``
    - Multimodal content lists with text parts

    Skips tool calls, system messages, and incomplete pairs. Messages
    whose role can't be identified are simply ignored rather than
    resetting the pending user message, so a stray unknown entry between
    a user message and the assistant reply doesn't break pairing.
    """
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None

    for msg in history:
        role = _msg_role(msg)
        text = _msg_text(msg)

        if role == "user":
            pending_user = text or None
            continue

        if role == "assistant":
            if pending_user is None:
                continue
            if text:
                pairs.append((pending_user, text))
                pending_user = None
                continue
            # Tool-call-only assistant messages (no text) -> keep waiting
            # for the next assistant message with actual content.
            if isinstance(msg, dict) and msg.get("tool_calls"):
                continue
            # Empty assistant message we can't pair; drop the pending user.
            pending_user = None
            continue

        # role in {"skip", "unknown"}: do nothing, keep pending_user.

    return pairs


def _describe_history_shape(history: list[dict], limit: int = 3) -> str:
    """Short string describing the first few history items' shape, used
    in /memory summarize debug output when pairs can't be extracted."""
    samples: list[str] = []
    for i, msg in enumerate(history[:limit]):
        if not isinstance(msg, dict):
            samples.append(f"#{i} type={type(msg).__name__}")
            continue
        keys = sorted(msg.keys())
        role_raw = (
            msg.get("role")
            or msg.get("type")
            or msg.get("sender_type")
            or msg.get("sender")
            or "?"
        )
        content = msg.get("content")
        content_type = type(content).__name__
        if isinstance(content, str):
            content_type = f"str({len(content)})"
        elif isinstance(content, list):
            content_type = f"list({len(content)})"
        samples.append(
            f"#{i} role={role_raw!r} content={content_type} keys={keys}"
        )
    return " | ".join(samples)


def _flatten_astrbot_content(parts: Any) -> str:
    if isinstance(parts, str):
        return parts.strip()
    if not isinstance(parts, list):
        return ""
    text_parts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue
        text = part.get("text", "")
        if isinstance(text, str) and text.strip():
            text_parts.append(text.strip())
    return "\n".join(text_parts).strip()


def _clean_import_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _RAG_BLOCK_RE.sub("", text)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    compact: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[发送时间:"):
            continue
        if stripped.startswith("[Image Attachment:"):
            continue
        if any(marker in stripped for marker in _ASTRBOT_SYSTEM_PROMPT_MARKERS):
            return ""
        compact.append(stripped)
    return "\n".join(compact).strip()


def _extract_pairs_from_astrbot_jsonl(raw_text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not raw_text.strip():
        return pairs

    for line in raw_text.splitlines()[:IMPORT_MAX_CONVERSATIONS]:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        messages = record.get("content")
        if not isinstance(messages, list):
            continue

        pending_user: str | None = None
        rounds = 0
        for msg in messages:
            if rounds >= IMPORT_MAX_ROUNDS_PER_CONVERSATION:
                break
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            text = _clean_import_text(_flatten_astrbot_content(msg.get("content")))
            if role == "user":
                pending_user = text or None
            elif role == "assistant" and pending_user and text:
                pairs.append((pending_user, text))
                pending_user = None
                rounds += 1

    return pairs
