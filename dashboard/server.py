"""Dashboard HTTP server: FastAPI app + uvicorn lifecycle.

Minimal viable dashboard with:
- Auth endpoints (setup / login / logout)
- Bucket list + detail + edit + delete API
- Search API
- Stats API
- Single-page HTML frontend (Chinese UI)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.embedding_service import pack_vector
from ..core.models import MemoryBucket, clamp_bucket
from .auth import AuthStore


_EXPORT_FORMAT = "astrbot-memory-dashboard-export-v1"
_BACKUP_PREFIX = "dashboard_import_backup_"
_CONFIG_EXPORT_NAME = "memory-dashboard-config.json"
_MEMORIES_EXPORT_NAME = "memory-dashboard-memories.zip"


def _split_dashboard_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for item in raw.split(","):
        s = item.strip()
        if s:
            out.append(s)
    return out


def _normalize_text_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        return _split_dashboard_tags(raw)
    return []


def _coerce_timestamp(value: Any) -> float | None:
    """Best-effort convert a dashboard-supplied date value to a unix
    timestamp (seconds). Accepts ``datetime-local`` strings (the format
    emitted by ``<input type="datetime-local">``), ISO 8601 strings and
    numeric values. Returns ``None`` for empty / unparseable inputs.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # ``datetime-local`` uses ``T`` as the separator; ``fromisoformat``
        # handles that natively. We also accept space-separated variants
        # for completeness.
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return None
    return None


def _bucket_payload(bucket, *, score: float | None = None) -> dict[str, Any]:
    payload = {
        "id": bucket.id,
        "session_id": bucket.session_id,
        "name": bucket.name,
        "content": bucket.content,
        "domain": bucket.domain,
        "tags": bucket.tags,
        "valence": bucket.valence,
        "arousal": bucket.arousal,
        "importance": bucket.importance,
        "bucket_type": bucket.bucket_type,
        "type": bucket.bucket_type,
        "pinned": bucket.pinned,
        "resolved": bucket.resolved,
        "digested": bucket.digested,
        "activation_count": bucket.activation_count,
        "created_at": bucket.created_at,
        "last_active_at": bucket.last_active_at,
        "created": bucket.created_at,
        "last_active": bucket.last_active_at,
    }
    if score is not None:
        rounded = round(score, 4)
        payload["score"] = rounded
    return payload


async def _pick_dashboard_session(plugin) -> str | None:
    """Pick which session the dashboard should display by default.

    Priority:
      1. ``dashboard_session_id`` config override (if user pinned one).
      2. The most-recently-active session, so users who jump between
         private and group chats land on the one they just used.

    The previous implementation fell back to the alphabetically-first
    session, which meant opening a brand-new chat with an earlier
    session_id could silently hide the user's existing memories from
    the dashboard view. The data was always intact in SQLite, but the
    user had no way to know without manually configuring
    ``dashboard_session_id``.
    """
    if plugin.manager is None:
        return None
    configured = str((getattr(plugin, "config", {}) or {}).get("dashboard_session_id", "")).strip()
    if configured:
        return configured
    sessions_meta = await plugin.manager.list_sessions_with_meta()
    if not sessions_meta:
        return None
    return str(sessions_meta[0]["session_id"])


if TYPE_CHECKING:
    from ..main import MemoryPlugin

logger = logging.getLogger("astrbot_plugin_ob_memory.dashboard")

# Session cookie name
COOKIE_NAME = "memory_dashboard_session"


class DashboardServer:
    """Embedded HTTP server for the memory dashboard."""

    def __init__(self, plugin: MemoryPlugin, data_dir: Path):
        self.plugin = plugin
        self.data_dir = data_dir
        self.auth = AuthStore(data_dir)
        self._task: asyncio.Task | None = None
        self._server: Any = None

    def _backup_path(self, name: str) -> Path:
        return self.data_dir / name

    def _iter_backup_files(self) -> list[tuple[str, str, str, Path]]:
        backups: list[tuple[str, str, str, Path]] = []
        for path in sorted(self.data_dir.glob(f"{_BACKUP_PREFIX}*.zip")):
            if path.is_file():
                backups.append(
                    (
                        path.name,
                        "file",
                        "导入记忆前自动生成的插件数据备份",
                        path,
                    )
                )
        return backups

    async def _list_all_buckets(self) -> list[MemoryBucket]:
        if self.plugin.manager is None:
            return []
        buckets: list[MemoryBucket] = []
        sessions = await self.plugin.manager.list_sessions()
        for session_id in sessions:
            buckets.extend(
                await self.plugin.manager.list_by_session(session_id, include_archived=True)
            )
        return buckets

    async def _export_config_payload(self) -> dict[str, Any]:
        return {
            "format": _EXPORT_FORMAT,
            "runtime_only": True,
            "config": {
                "dehydration": {
                    "model": str(self.plugin.config.get("digest_model", "") or ""),
                    "prompt": str(self.plugin.config.get("digest_prompt", "") or ""),
                },
                "embedding": {
                    "provider_id": str(
                        self.plugin.config.get("embedding_provider_id", "") or ""
                    ),
                },
                "tagging_enabled": bool(self.plugin.config.get("tagging_enabled", True)),
                "merge_threshold": float(self.plugin.config.get("merge_threshold", 0.85)),
            },
        }

    async def _export_memories_bytes(self) -> bytes:
        buckets = await self._list_all_buckets()
        memories = [_bucket_payload(bucket) for bucket in buckets]
        embeddings: list[dict[str, Any]] = []
        if self.plugin.embedding is not None:
            for bucket in buckets:
                vec = await self.plugin.embedding.get(bucket.id)
                if vec:
                    embeddings.append({"bucket_id": bucket.id, "dim": len(vec), "vector": vec})

        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "format": _EXPORT_FORMAT,
                        "kind": "memories",
                        "exported_at": int(time.time()),
                        "runtime_only": True,
                    },
                    ensure_ascii=False,
                ),
            )
            zf.writestr(
                "memories.json",
                json.dumps({"memories": memories}, ensure_ascii=False),
            )
            zf.writestr(
                "embeddings.json",
                json.dumps({"embeddings": embeddings}, ensure_ascii=False),
            )
        return payload.getvalue()

    async def _parse_import_zip(self, raw: bytes) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw), "r")
        except zipfile.BadZipFile as e:
            raise ValueError(f"zip 解析失败: {e}") from e

        with archive:
            names = set(archive.namelist())
            if "manifest.json" not in names or "memories.json" not in names:
                raise ValueError("zip 缺少 manifest.json 或 memories.json")
            try:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                memories_doc = json.loads(archive.read("memories.json").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise ValueError(f"zip 内容解析失败: {e}") from e
            embeddings_doc: dict[str, Any] = {"embeddings": []}
            if "embeddings.json" in names:
                try:
                    embeddings_doc = json.loads(archive.read("embeddings.json").decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    raise ValueError(f"embeddings.json 解析失败: {e}") from e

        if manifest.get("format") != _EXPORT_FORMAT:
            raise ValueError("不支持的导入格式")
        memories = memories_doc.get("memories")
        embeddings = embeddings_doc.get("embeddings") or []
        if not isinstance(memories, list):
            raise ValueError("memories.json 缺少 memories 列表")
        if not isinstance(embeddings, list):
            raise ValueError("embeddings.json 缺少 embeddings 列表")
        return manifest, memories, embeddings

    def _memory_bucket_from_payload(self, item: dict[str, Any]) -> MemoryBucket:
        if not isinstance(item, dict):
            raise ValueError("记忆条目格式错误")
        session_id = str(item.get("session_id", "") or "").strip()
        bucket_id = str(item.get("id", "") or "").strip()
        content = str(item.get("content", "") or "")
        if not session_id or not bucket_id or not content:
            raise ValueError("记忆条目缺少 id / session_id / content")
        bucket = MemoryBucket(
            id=bucket_id,
            session_id=session_id,
            name=str(item.get("name", "") or ""),
            content=content,
            domain=_normalize_text_list(item.get("domain")),
            tags=_normalize_text_list(item.get("tags")),
            valence=item.get("valence", 0.5),
            arousal=item.get("arousal", 0.3),
            importance=item.get("importance", 5),
            bucket_type=str(item.get("bucket_type") or item.get("type") or "dynamic"),
            pinned=bool(item.get("pinned", False)),
            resolved=bool(item.get("resolved", False)),
            digested=bool(item.get("digested", False)),
            activation_count=float(item.get("activation_count", 0.0) or 0.0),
            created_at=float(item.get("created_at", item.get("created", 0.0)) or 0.0),
            last_active_at=float(
                item.get("last_active_at", item.get("last_active", 0.0)) or 0.0
            ),
        )
        return clamp_bucket(bucket)

    async def _create_dashboard_backup(self, raw: bytes) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self._backup_path(f"{_BACKUP_PREFIX}{ts}.zip")
        path.write_bytes(raw)
        return path.name

    async def _restore_from_backup(self, backup_name: str) -> None:
        path = self._backup_path(backup_name)
        raw = path.read_bytes()
        await self._replace_from_import_bytes(raw)

    async def _replace_from_import_bytes(self, raw: bytes) -> tuple[int, int]:
        _manifest, memories, embeddings = await self._parse_import_zip(raw)
        if self.plugin.db is None:
            raise RuntimeError("未初始化")
        async with self.plugin.db.transaction() as conn:
            await conn.execute("DELETE FROM embeddings")
            await conn.execute("DELETE FROM memories")
            for item in memories:
                bucket = self._memory_bucket_from_payload(item)
                await conn.execute(
                    "INSERT INTO memories (id, session_id, name, content, domain, tags, valence, arousal, importance, bucket_type, pinned, resolved, digested, model_valence, source_bucket_id, activation_count, created_at, last_active_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        bucket.id,
                        bucket.session_id,
                        bucket.name,
                        bucket.content,
                        json.dumps(bucket.domain, ensure_ascii=False),
                        json.dumps(bucket.tags, ensure_ascii=False),
                        float(bucket.valence),
                        float(bucket.arousal),
                        int(bucket.importance),
                        bucket.bucket_type,
                        1 if bucket.pinned else 0,
                        1 if bucket.resolved else 0,
                        1 if bucket.digested else 0,
                        bucket.model_valence,
                        bucket.source_bucket_id,
                        float(bucket.activation_count),
                        float(bucket.created_at),
                        float(bucket.last_active_at),
                    ),
                )
            for item in embeddings:
                if not isinstance(item, dict):
                    continue
                bucket_id = str(item.get("bucket_id", "") or "").strip()
                vector = item.get("vector")
                if not bucket_id or not isinstance(vector, list):
                    continue
                try:
                    dim = len(vector)
                    packed = pack_vector([float(v) for v in vector])
                except (TypeError, ValueError, AttributeError):
                    continue
                await conn.execute(
                    "INSERT OR REPLACE INTO embeddings (bucket_id, vector, dim, updated_at) VALUES (?, ?, ?, ?)",
                    (bucket_id, packed, dim, time.time()),
                )
        return len(memories), len(embeddings)

    def build_app(self):
        """Create the Starlette/ASGI app with all routes."""
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
        from starlette.routing import Route

        # ------------------------------------------------------------------
        # Auth helpers
        # ------------------------------------------------------------------
        def _get_token(request: Request) -> str | None:
            return request.cookies.get(COOKIE_NAME)

        def _require_auth(request: Request) -> JSONResponse | None:
            token = _get_token(request)
            if not self.auth.validate_session(token):
                return JSONResponse({"detail": "auth required"}, status_code=401)
            return None

        # ------------------------------------------------------------------
        # Auth endpoints
        # ------------------------------------------------------------------
        async def auth_status(request: Request) -> Response:
            token = _get_token(request)
            return JSONResponse(
                {
                    **self.auth.to_status_dict(),
                    "authenticated": self.auth.validate_session(token),
                }
            )

        async def auth_setup(request: Request) -> Response:
            if not self.auth.setup_needed:
                return JSONResponse({"error": "已配置"}, status_code=400)
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "无效请求"}, status_code=400)
            password = body.get("password", "")
            if not self.auth.setup_password(password):
                return JSONResponse({"error": "密码至少4位"}, status_code=400)
            token = self.auth.create_session()
            resp = JSONResponse({"ok": True})
            resp.set_cookie(
                COOKIE_NAME, token, httponly=True, samesite="lax", max_age=7 * 86400
            )
            return resp

        async def auth_login(request: Request) -> Response:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "无效请求"}, status_code=400)
            password = body.get("password", "")
            if self.auth.check_password(password):
                token = self.auth.create_session()
                resp = JSONResponse({"ok": True})
                resp.set_cookie(
                    COOKIE_NAME, token, httponly=True, samesite="lax", max_age=7 * 86400
                )
                return resp
            return JSONResponse({"error": "密码错误"}, status_code=401)

        async def auth_logout(request: Request) -> Response:
            self.auth.revoke_session(_get_token(request))
            resp = JSONResponse({"ok": True})
            resp.delete_cookie(COOKIE_NAME)
            return resp

        async def auth_change_password(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.auth.env_locked:
                return JSONResponse(
                    {"error": "密码由环境变量控制，无法在此修改"}, status_code=400
                )
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "无效请求"}, status_code=400)
            current = body.get("current", "")
            new_pwd = body.get("new", "")
            if self.auth.change_password(current, new_pwd):
                token = self.auth.create_session()
                resp = JSONResponse({"ok": True})
                resp.set_cookie(
                    COOKIE_NAME, token, httponly=True, samesite="lax", max_age=7 * 86400
                )
                return resp
            return JSONResponse({"error": "当前密码错误或新密码太短"}, status_code=400)

        # ------------------------------------------------------------------
        # Health
        # ------------------------------------------------------------------
        async def health(request: Request) -> Response:
            return JSONResponse(
                {
                    "status": "ok",
                    "version": "0.1.0",
                }
            )

        # ------------------------------------------------------------------
        # API endpoints (require auth)
        # ------------------------------------------------------------------
        async def api_buckets(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.manager is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)

            session_filter = request.query_params.get("session", "")
            type_filter = request.query_params.get("type", "")
            q = request.query_params.get("q", "")
            try:
                limit = int(request.query_params.get("limit", "100"))
            except ValueError:
                limit = 100

            # Get all sessions or filter by one
            from ..core.decay_engine import calculate_score

            if session_filter:
                sessions = [session_filter]
            else:
                sessions = await self.plugin.manager.list_sessions()

            all_buckets = []
            for sid in sessions:
                buckets = await self.plugin.manager.list_by_session(
                    sid, include_archived=True
                )
                for b in buckets:
                    if type_filter and b.bucket_type != type_filter:
                        continue
                    if (
                        q
                        and q.lower()
                        not in (
                            (b.name or "")
                            + " "
                            + (b.content or "")
                            + " "
                            + " ".join(b.tags)
                        ).lower()
                    ):
                        continue
                    score = calculate_score(b)
                    all_buckets.append(
                        {
                            "id": b.id,
                            "session_id": b.session_id,
                            "name": b.name,
                            "domain": b.domain,
                            "tags": b.tags,
                            "valence": b.valence,
                            "arousal": b.arousal,
                            "importance": b.importance,
                            "bucket_type": b.bucket_type,
                            "pinned": b.pinned,
                            "resolved": b.resolved,
                            "digested": b.digested,
                            "activation_count": b.activation_count,
                            "score": round(score, 4),
                            "created_at": b.created_at,
                            "last_active_at": b.last_active_at,
                            "content_preview": (b.content or "")[:200],
                        }
                    )

            all_buckets.sort(key=lambda x: x["score"], reverse=True)
            return JSONResponse(
                {
                    "total": len(all_buckets),
                    "buckets": all_buckets[:limit],
                }
            )

        async def api_bucket_detail(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.manager is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)

            bucket_id = request.path_params["bucket_id"]
            # Find across all sessions
            sessions = await self.plugin.manager.list_sessions()
            for sid in sessions:
                b = await self.plugin.manager.get(sid, bucket_id)
                if b is not None:
                    from ..core.decay_engine import calculate_score

                    score = calculate_score(b)
                    return JSONResponse(
                        {
                            "id": b.id,
                            "session_id": b.session_id,
                            "name": b.name,
                            "content": b.content,
                            "domain": b.domain,
                            "tags": b.tags,
                            "valence": b.valence,
                            "arousal": b.arousal,
                            "importance": b.importance,
                            "bucket_type": b.bucket_type,
                            "pinned": b.pinned,
                            "resolved": b.resolved,
                            "digested": b.digested,
                            "model_valence": b.model_valence,
                            "source_bucket_id": b.source_bucket_id,
                            "activation_count": b.activation_count,
                            "score": round(score, 4),
                            "created_at": b.created_at,
                            "last_active_at": b.last_active_at,
                        }
                    )
            return JSONResponse({"error": "未找到"}, status_code=404)

        async def api_bucket_update(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.manager is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)

            bucket_id = request.path_params["bucket_id"]
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "无效请求"}, status_code=400)

            # Find the bucket's session
            sessions = await self.plugin.manager.list_sessions()
            target_session = None
            for sid in sessions:
                b = await self.plugin.manager.get(sid, bucket_id)
                if b is not None:
                    target_session = sid
                    break
            if target_session is None:
                return JSONResponse({"error": "未找到"}, status_code=404)

            # Extract updatable fields
            fields: dict[str, Any] = {}
            for key in (
                "name",
                "content",
                "valence",
                "arousal",
                "importance",
                "pinned",
                "resolved",
                "digested",
            ):
                if key in body:
                    fields[key] = body[key]
            if "domain" in body:
                fields["domain"] = (
                    body["domain"]
                    if isinstance(body["domain"], list)
                    else [body["domain"]]
                )
            if "tags" in body:
                fields["tags"] = (
                    body["tags"]
                    if isinstance(body["tags"], list)
                    else _split_dashboard_tags(body["tags"])
                )

            if not fields:
                return JSONResponse({"error": "无更新字段"}, status_code=400)

            updated = await self.plugin.manager.update(
                target_session, bucket_id, **fields
            )
            if updated is None:
                return JSONResponse({"error": "更新失败"}, status_code=500)
            if (
                "content" in fields
                and self.plugin.embedding is not None
                and self.plugin.embedding.enabled
            ):
                try:
                    await self.plugin.embedding.generate_and_store(
                        bucket_id, updated.content
                    )
                except Exception as e:
                    logger.warning(
                        "dashboard embedding refresh failed for %s: %s", bucket_id, e
                    )
            return JSONResponse({"ok": True, "id": bucket_id})

        async def api_bucket_delete(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.manager is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)

            bucket_id = request.path_params["bucket_id"]
            sessions = await self.plugin.manager.list_sessions()
            for sid in sessions:
                b = await self.plugin.manager.get(sid, bucket_id)
                if b is not None:
                    ok = await self.plugin.manager.delete(sid, bucket_id)
                    if ok:
                        return JSONResponse({"ok": True})
                    return JSONResponse({"error": "删除失败"}, status_code=500)
            return JSONResponse({"error": "未找到"}, status_code=404)

        async def api_sessions(request: Request) -> Response:
            """List all sessions with memory counts and recency.

            Returned shape:

                {
                  "sessions": [
                    {"session_id": "...", "memory_count": int,
                     "last_active_at": float},
                    ...
                  ],
                  "current": "..."   # the session the dashboard would
                                     # default to right now
                }

            Powers the dashboard's session-switcher dropdown so users
            can flip between e.g. their private chat and a group chat
            without editing the ``dashboard_session_id`` config by hand.
            """
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.manager is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)

            sessions_meta = await self.plugin.manager.list_sessions_with_meta()
            current = await _pick_dashboard_session(self.plugin)
            return JSONResponse({"sessions": sessions_meta, "current": current})

        async def api_stats(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.manager is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)

            sessions = await self.plugin.manager.list_sessions()
            total_counts: dict[str, int] = {}
            now_ts = time.time()
            today_start = datetime.fromtimestamp(now_ts).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp()
            week_start = now_ts - 7 * 86400
            today_new = 0
            week_new = 0
            max_activation = 0.0
            for sid in sessions:
                counts = await self.plugin.manager.count_in_session(sid)
                for k, v in counts.items():
                    total_counts[k] = total_counts.get(k, 0) + v
                buckets = await self.plugin.manager.list_by_session(sid, include_archived=True)
                for bucket in buckets:
                    if bucket.created_at >= today_start:
                        today_new += 1
                    if bucket.created_at >= week_start:
                        week_new += 1
                    max_activation = max(max_activation, float(bucket.activation_count or 0.0))

            decay_status = "未运行"
            if self.plugin.decay is not None:
                decay_status = "运行中" if self.plugin.decay.is_running else "已停止"

            embedding_status = "未启用"
            if self.plugin.embedding is not None and self.plugin.embedding.enabled:
                embedding_status = "已启用"

            return JSONResponse(
                {
                    "sessions": len(sessions),
                    "counts": total_counts,
                    "total": sum(total_counts.values()),
                    "today_new": today_new,
                    "week_new": week_new,
                    "max_activation": round(max_activation, 1),
                    "decay_engine": decay_status,
                    "embedding": embedding_status,
                }
            )

        async def api_search(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.search is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)

            q = request.query_params.get("q", "")
            session_id = request.query_params.get("session", "")
            if not q:
                return JSONResponse({"error": "缺少 q 参数"}, status_code=400)
            if not session_id:
                # Use first available session
                if self.plugin.manager:
                    sessions = await self.plugin.manager.list_sessions()
                    session_id = sessions[0] if sessions else ""
            if not session_id:
                return JSONResponse({"results": []})

            try:
                hits = await self.plugin.search.search(session_id, q, limit=10)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

            results = []
            for h in hits:
                b = h.bucket
                results.append(
                    {
                        "id": b.id,
                        "name": b.name,
                        "score": round(h.score, 2),
                        "via": h.via,
                        "domain": b.domain,
                        "content_preview": (b.content or "")[:200],
                        "importance": b.importance,
                        "resolved": b.resolved,
                        "pinned": b.pinned,
                    }
                )
            return JSONResponse({"results": results})

        async def api_memories(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.manager is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)

            session_id = request.query_params.get("session") or await _pick_dashboard_session(
                self.plugin
            )
            if not session_id:
                return JSONResponse([])

            from ..core.decay_engine import calculate_score

            buckets = await self.plugin.manager.list_by_session(
                session_id, include_archived=True
            )
            buckets.sort(key=calculate_score, reverse=True)
            payload = []
            for bucket in buckets:
                item = _bucket_payload(bucket, score=calculate_score(bucket))
                item["content_preview"] = (bucket.content or "")[:200]
                payload.append(item)
            return JSONResponse(payload)

        async def api_memory_create(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.writer is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)

            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "无效请求"}, status_code=400)

            content = str(body.get("content", "") or "").strip()
            if not content:
                return JSONResponse({"error": "正文不能为空"}, status_code=400)

            session_id = str(body.get("session_id", "") or "").strip() or await _pick_dashboard_session(
                self.plugin
            )
            if not session_id:
                session_id = "dashboard"

            analysis_payload = body.get("analysis")
            if not isinstance(analysis_payload, dict):
                analysis_payload = None

            try:
                if analysis_payload is not None:
                    result = await self.plugin.writer._persist_with_analysis(
                        session_id,
                        content,
                        analysis=analysis_payload,
                        importance=body.get("importance"),
                        tags=_normalize_text_list(body.get("tags")),
                        pinned=bool(body.get("pinned", False)),
                        valence=body.get("valence"),
                        arousal=body.get("arousal"),
                    )
                else:
                    result = await self.plugin.writer.hold(
                        session_id,
                        content,
                        importance=body.get("importance"),
                        tags=_normalize_text_list(body.get("tags")),
                        pinned=bool(body.get("pinned", False)),
                        valence=body.get("valence"),
                        arousal=body.get("arousal"),
                    )
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

            bucket = result.target_bucket
            updates: dict[str, Any] = {}
            name = str(body.get("name", "") or "").strip()
            if name:
                updates["name"] = name
            domain = _normalize_text_list(body.get("domain"))
            if domain:
                updates["domain"] = domain
            # Honour a user-supplied creation date for manual inscription.
            # Skip when the writer merged into an existing bucket — we don't
            # want to silently overwrite the older memory's timestamp.
            if not getattr(result, "was_merged", False):
                created_raw = body.get("created", body.get("created_at"))
                created_ts = _coerce_timestamp(created_raw)
                if created_ts is not None and abs(created_ts - bucket.created_at) > 1.0:
                    updates["created_at"] = created_ts
                    # Align last_active_at so the new memory doesn't show up
                    # as "just touched" in surface / decay calculations.
                    updates["last_active_at"] = created_ts
            if updates:
                updated = await self.plugin.manager.update(session_id, bucket.id, **updates)
                if updated is not None:
                    bucket = updated
            return JSONResponse(_bucket_payload(bucket), status_code=201)

        async def api_memory_update(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.manager is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)

            bucket_id = request.path_params["bucket_id"]
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "无效请求"}, status_code=400)

            session_id = str(body.get("session_id", "") or "").strip()
            target = None
            target_session = session_id or None
            if target_session:
                target = await self.plugin.manager.get(target_session, bucket_id)
            else:
                sessions = await self.plugin.manager.list_sessions()
                for sid in sessions:
                    target = await self.plugin.manager.get(sid, bucket_id)
                    if target is not None:
                        target_session = sid
                        break
            if target is None or target_session is None:
                return JSONResponse({"error": "未找到该记忆"}, status_code=404)

            fields: dict[str, Any] = {}
            for key in (
                "name",
                "content",
                "valence",
                "arousal",
                "importance",
                "pinned",
                "resolved",
                "digested",
                "activation_count",
                "last_active_at",
                "created_at",
            ):
                if key in body:
                    fields[key] = body[key]
            if "last_active" in body and "last_active_at" not in fields:
                fields["last_active_at"] = body["last_active"]
            if "created" in body and "created_at" not in fields:
                ts = _coerce_timestamp(body["created"])
                if ts is not None:
                    fields["created_at"] = ts
            elif "created_at" in fields:
                ts = _coerce_timestamp(fields["created_at"])
                if ts is None:
                    fields.pop("created_at", None)
                else:
                    fields["created_at"] = ts
            if "last_active_at" in fields:
                ts = _coerce_timestamp(fields["last_active_at"])
                if ts is None:
                    fields.pop("last_active_at", None)
                else:
                    fields["last_active_at"] = ts
            if "domain" in body:
                fields["domain"] = _normalize_text_list(body.get("domain"))
            if "tags" in body:
                fields["tags"] = _normalize_text_list(body.get("tags"))
            if not fields:
                return JSONResponse({"error": "无更新字段"}, status_code=400)

            updated = await self.plugin.manager.update(target_session, bucket_id, **fields)
            if updated is None:
                return JSONResponse({"error": "更新失败"}, status_code=500)
            if (
                "content" in fields
                and self.plugin.embedding is not None
                and self.plugin.embedding.enabled
            ):
                try:
                    await self.plugin.embedding.generate_and_store(bucket_id, updated.content)
                except Exception as e:
                    logger.warning(
                        "dashboard embedding refresh failed for %s: %s", bucket_id, e
                    )
            return JSONResponse(_bucket_payload(updated))

        async def api_memory_delete(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.manager is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)

            bucket_id = request.path_params["bucket_id"]
            sessions = await self.plugin.manager.list_sessions()
            for sid in sessions:
                if await self.plugin.manager.delete(sid, bucket_id):
                    return JSONResponse({"ok": True})
            return JSONResponse({"error": "未找到该记忆"}, status_code=404)

        async def api_analyze(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.tagger is None:
                return JSONResponse({
                    "domain": ["未分类"],
                    "valence": 0.5,
                    "arousal": 0.3,
                    "tags": [],
                    "suggested_name": "",
                    "importance": 5,
                })
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "无效请求"}, status_code=400)
            content = str(body.get("content", "") or "").strip()
            if not content:
                return JSONResponse({
                    "domain": ["未分类"],
                    "valence": 0.5,
                    "arousal": 0.3,
                    "tags": [],
                    "suggested_name": "",
                    "importance": 5,
                })
            try:
                session_id = await _pick_dashboard_session(self.plugin)
                analysis = await self.plugin.tagger.analyze(content, session_id=session_id)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
            return JSONResponse(analysis)

        async def api_grow(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.tagger is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "无效请求"}, status_code=400)
            content = str(body.get("content", "") or "").strip()
            if not content:
                return JSONResponse({"error": "内容不能为空"}, status_code=400)
            try:
                session_id = await _pick_dashboard_session(self.plugin)
                entries = await self.plugin.tagger.digest(content, session_id=session_id)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
            return JSONResponse(entries[:10])

        async def api_status_compat(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            model = str((getattr(self.plugin.writer, "digest_prompt", "") or "")).strip()
            return JSONResponse(
                {
                    "ai_available": self.plugin.tagger is not None,
                    "model": model or None,
                }
            )

        async def api_config_get(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            cfg = getattr(self.plugin, "config", {}) or {}
            return JSONResponse(
                {
                    "dehydration": {
                        "api_key": str(cfg.get("dashboard_dehydration_api_key", "") or ""),
                        "base_url": str(cfg.get("dashboard_dehydration_base_url", "") or ""),
                        "model": str(cfg.get("digest_model", "") or ""),
                        "prompt": str(cfg.get("digest_prompt", "") or ""),
                    },
                    "embedding": {
                        "provider_id": str(cfg.get("embedding_provider_id", "") or ""),
                        "api_key": str(cfg.get("dashboard_embedding_api_key", "") or ""),
                        "base_url": str(cfg.get("dashboard_embedding_base_url", "") or ""),
                        "model": str(cfg.get("dashboard_embedding_model", "") or ""),
                    },
                    "tagging_enabled": bool(cfg.get("tagging_enabled", True)),
                    "merge_threshold": float(cfg.get("merge_threshold", 0.85)),
                }
            )

        async def api_config_put(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "无效请求"}, status_code=400)
            updated: dict[str, Any] = {}
            if "tagging_enabled" in body:
                updated["tagging_enabled"] = bool(body["tagging_enabled"])
            if "merge_threshold" in body:
                updated["merge_threshold"] = body["merge_threshold"]
            dehydration = body.get("dehydration")
            if isinstance(dehydration, dict):
                if "api_key" in dehydration:
                    updated["dashboard_dehydration_api_key"] = str(
                        dehydration.get("api_key", "") or ""
                    ).strip()
                if "base_url" in dehydration:
                    updated["dashboard_dehydration_base_url"] = str(
                        dehydration.get("base_url", "") or ""
                    ).strip()
                if "prompt" in dehydration:
                    updated["digest_prompt"] = str(dehydration.get("prompt", "") or "").strip()
                if "model" in dehydration:
                    updated["digest_model"] = str(dehydration.get("model", "") or "").strip()
            embedding = body.get("embedding")
            if isinstance(embedding, dict):
                if "provider_id" in embedding:
                    updated["embedding_provider_id"] = str(
                        embedding.get("provider_id", "") or ""
                    ).strip()
                if "api_key" in embedding:
                    updated["dashboard_embedding_api_key"] = str(
                        embedding.get("api_key", "") or ""
                    ).strip()
                if "base_url" in embedding:
                    updated["dashboard_embedding_base_url"] = str(
                        embedding.get("base_url", "") or ""
                    ).strip()
                if "model" in embedding:
                    updated["dashboard_embedding_model"] = str(
                        embedding.get("model", "") or ""
                    ).strip()
            if updated:
                self.plugin.config.update(updated)
                if self.plugin.writer is not None:
                    if "tagging_enabled" in updated:
                        self.plugin.writer.tagging_enabled = bool(updated["tagging_enabled"])
                        self.plugin.writer.merge_enabled = bool(updated["tagging_enabled"])
                    if "merge_threshold" in updated:
                        try:
                            self.plugin.writer.merge_threshold = float(updated["merge_threshold"])
                        except (TypeError, ValueError):
                            pass
                    if "digest_prompt" in updated:
                        self.plugin.writer.digest_prompt = str(updated["digest_prompt"])
            return JSONResponse({"ok": True, "runtime_only": True})

        async def api_backfill_embeddings(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.plugin.manager is None or self.plugin.embedding is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)
            if not self.plugin.embedding.enabled:
                return JSONResponse({"error": "embedding 未启用"}, status_code=400)
            try:
                body = await request.json()
            except Exception:
                body = {}
            dry_run = bool(body.get("dry_run", False))
            buckets = await self._list_all_buckets()
            missing: list[MemoryBucket] = []
            for bucket in buckets:
                if not (bucket.content or "").strip():
                    continue
                if await self.plugin.embedding.get(bucket.id) is None:
                    missing.append(bucket)
            if dry_run:
                return JSONResponse(
                    {
                        "dry_run": True,
                        "total": len(buckets),
                        "have": len(buckets) - len(missing),
                        "missing": len(missing),
                        "sample": [
                            {"id": bucket.id, "name": bucket.name, "session_id": bucket.session_id}
                            for bucket in missing[:10]
                        ],
                    }
                )
            success = 0
            failed = 0
            for bucket in missing:
                try:
                    ok = await self.plugin.embedding.generate_and_store(bucket.id, bucket.content)
                except Exception:
                    ok = False
                if ok:
                    success += 1
                else:
                    failed += 1
            return JSONResponse(
                {
                    "dry_run": False,
                    "total": len(buckets),
                    "missing_before": len(missing),
                    "success": success,
                    "failed": failed,
                }
            )

        async def api_export_config(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            payload = await self._export_config_payload()
            return JSONResponse(payload, headers={"content-disposition": f'attachment; filename="{_CONFIG_EXPORT_NAME}"'})

        async def api_export_memories(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            payload = await self._export_memories_bytes()
            return Response(
                payload,
                media_type="application/zip",
                headers={"content-disposition": f'attachment; filename="{_MEMORIES_EXPORT_NAME}"'},
            )

        async def api_import_config(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                return JSONResponse({"error": "未上传文件"}, status_code=400)
            try:
                raw = await upload.read()
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                return JSONResponse({"error": "配置文件解析失败"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse({"error": "配置文件格式错误"}, status_code=400)
            payload = body.get("config") if body.get("format") == _EXPORT_FORMAT else body
            if not isinstance(payload, dict):
                return JSONResponse({"error": "配置内容格式错误"}, status_code=400)
            fake_request_body = {
                "dehydration": payload.get("dehydration", {}),
                "embedding": payload.get("embedding", {}),
                "tagging_enabled": payload.get("tagging_enabled", True),
                "merge_threshold": payload.get("merge_threshold", 0.85),
            }
            self.plugin.config.update(
                {
                    "digest_model": str(
                        ((fake_request_body.get("dehydration") or {}).get("model", "")) or ""
                    ).strip(),
                    "digest_prompt": str(
                        ((fake_request_body.get("dehydration") or {}).get("prompt", "")) or ""
                    ).strip(),
                    "embedding_provider_id": str(
                        ((fake_request_body.get("embedding") or {}).get("provider_id", "")) or ""
                    ).strip(),
                    "tagging_enabled": bool(fake_request_body.get("tagging_enabled", True)),
                    "merge_threshold": fake_request_body.get("merge_threshold", 0.85),
                }
            )
            if self.plugin.writer is not None:
                self.plugin.writer.tagging_enabled = bool(self.plugin.config.get("tagging_enabled", True))
                self.plugin.writer.merge_enabled = bool(self.plugin.config.get("tagging_enabled", True))
                try:
                    self.plugin.writer.merge_threshold = float(
                        self.plugin.config.get("merge_threshold", 0.85)
                    )
                except (TypeError, ValueError):
                    pass
                self.plugin.writer.digest_prompt = str(
                    self.plugin.config.get("digest_prompt", "") or ""
                )
            return JSONResponse({"ok": True, "runtime_only": True})

        async def api_import_memories(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                return JSONResponse({"error": "未上传文件"}, status_code=400)
            mode = str(form.get("mode", "dry_run") or "dry_run").strip().lower()
            if mode not in {"dry_run", "merge", "replace"}:
                return JSONResponse({"error": "未知 mode"}, status_code=400)
            try:
                raw = await upload.read()
                _manifest, memories, _embeddings = await self._parse_import_zip(raw)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            local_buckets = await self._list_all_buckets()
            local_ids = {bucket.id for bucket in local_buckets}
            import_ids = {
                str(item.get("id", "") or "").strip()
                for item in memories
                if isinstance(item, dict)
            }
            import_ids.discard("")
            new_ids = sorted(import_ids - local_ids)
            conflict_ids = sorted(import_ids & local_ids)
            only_local_ids = sorted(local_ids - import_ids)
            summary = {
                "mode": mode,
                "import_total": len(import_ids),
                "local_total": len(local_ids),
                "new_count": len(new_ids),
                "conflict_count": len(conflict_ids),
                "only_local_count": len(only_local_ids),
            }
            if mode == "dry_run":
                return JSONResponse(
                    {
                        **summary,
                        "dry_run": True,
                        "new_sample": new_ids[:10],
                        "conflict_sample": conflict_ids[:10],
                    }
                )
            backup_name = await self._create_dashboard_backup(await self._export_memories_bytes())
            if mode == "replace":
                try:
                    imported_count, imported_embeddings = await self._replace_from_import_bytes(raw)
                except Exception as e:
                    await self._restore_from_backup(backup_name)
                    return JSONResponse({"error": f"导入失败: {e}"}, status_code=500)
                return JSONResponse(
                    {
                        **summary,
                        "ok": True,
                        "backup": backup_name,
                        "imported": imported_count,
                        "embeddings_imported": imported_embeddings,
                    }
                )
            added = 0
            if self.plugin.manager is None:
                return JSONResponse({"error": "未初始化"}, status_code=503)
            for item in memories:
                if not isinstance(item, dict):
                    continue
                bucket_id = str(item.get("id", "") or "").strip()
                if not bucket_id or bucket_id in local_ids:
                    continue
                bucket = self._memory_bucket_from_payload(item)
                await self.plugin.manager.create(bucket)
                added += 1
            return JSONResponse(
                {
                    **summary,
                    "ok": True,
                    "added": added,
                    "backup": backup_name,
                    "note": "导入已完成；若有缺失向量，可再执行补建。",
                }
            )

        async def api_backups_list(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            items = []
            total_size = 0
            for name, kind, category, path in self._iter_backup_files():
                stat = path.stat()
                size = stat.st_size
                total_size += size
                items.append(
                    {
                        "name": name,
                        "kind": kind,
                        "category": category,
                        "size_bytes": size,
                        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                        "mtime_ts": stat.st_mtime,
                    }
                )
            items.sort(key=lambda item: item["mtime_ts"], reverse=True)
            return JSONResponse(
                {"items": items, "count": len(items), "total_size_bytes": total_size}
            )

        async def api_backups_delete(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "无效请求"}, status_code=400)
            name = str(body.get("name", "") or "").strip()
            if not name:
                return JSONResponse({"error": "缺少 name 参数"}, status_code=400)
            if "/" in name or "\\" in name or ".." in name:
                return JSONResponse({"error": "name 不能含路径分隔符或 .."}, status_code=400)
            path = self._backup_path(name)
            if path.parent != self.data_dir or not path.name.startswith(_BACKUP_PREFIX):
                return JSONResponse({"error": "不匹配任何已知备份 pattern"}, status_code=400)
            if not path.exists() or not path.is_file():
                return JSONResponse({"error": f"{name} 不存在"}, status_code=404)
            size = path.stat().st_size
            path.unlink()
            return JSONResponse({"ok": True, "deleted": name, "freed_bytes": size})

        async def api_session_status(request: Request) -> Response:
            token = _get_token(request)
            return JSONResponse({"authenticated": self.auth.validate_session(token)})

        async def api_password_put(request: Request) -> Response:
            err = _require_auth(request)
            if err:
                return err
            if self.auth.env_locked:
                return JSONResponse(
                    {"error": "密码由环境变量控制，无法在此修改"}, status_code=400
                )
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "无效请求"}, status_code=400)
            current = body.get("old_password")
            if current is None:
                current = body.get("current", "")
            new_pwd = body.get("new_password")
            if new_pwd is None:
                new_pwd = body.get("new", "")
            if self.auth.change_password(current, new_pwd):
                return JSONResponse({"ok": True})
            return JSONResponse({"error": "原密码错误或新密码太短"}, status_code=400)

        # ------------------------------------------------------------------
        # Frontend
        # ------------------------------------------------------------------
        async def dashboard_page(request: Request) -> Response:
            html_path = Path(__file__).parent / "static" / "index.html"
            if not html_path.exists():
                return HTMLResponse("<h1>Dashboard 前端文件缺失</h1>", status_code=500)
            return HTMLResponse(html_path.read_text(encoding="utf-8"))

        async def root_page(request: Request) -> Response:
            """Convenience: / also serves the dashboard directly."""
            return await dashboard_page(request)

        async def manifest(request: Request) -> Response:
            file_path = Path(__file__).parent / "static" / "manifest.json"
            if not file_path.exists():
                return Response(status_code=404)
            return FileResponse(file_path, media_type="application/manifest+json")

        async def icon_file(request: Request) -> Response:
            filename = request.path_params["filename"]
            if ".." in filename or "\\" in filename:
                return Response(status_code=404)
            static_dir = Path(__file__).parent / "static"
            candidates = [
                static_dir / filename,
                static_dir / f"{Path(filename).stem}.svg",
            ]
            for file_path in candidates:
                if file_path.exists() and file_path.is_file():
                    return FileResponse(file_path)
            return Response(status_code=404)

        async def static_file(request: Request) -> Response:
            filename = request.path_params["filename"]
            if ".." in filename or "/" in filename or "\\" in filename:
                return Response(status_code=404)
            static_dir = Path(__file__).parent / "static"
            file_path = static_dir / filename
            if not file_path.exists() or not file_path.is_file():
                return Response(status_code=404)
            media_type = {
                ".css": "text/css",
                ".js": "application/javascript",
                ".json": "application/json",
                ".html": "text/html",
                ".png": "image/png",
                ".svg": "image/svg+xml",
                ".webmanifest": "application/manifest+json",
            }.get(file_path.suffix.lower())
            return FileResponse(file_path, media_type=media_type)

        # ------------------------------------------------------------------
        # Build app
        # ------------------------------------------------------------------
        routes = [
            Route("/", root_page, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
            Route("/dashboard", dashboard_page, methods=["GET"]),
            Route("/manifest.json", manifest, methods=["GET"]),
            Route("/icons/{filename:path}", icon_file, methods=["GET"]),
            Route("/static/{filename}", static_file, methods=["GET"]),
            # Auth
            Route("/auth/status", auth_status, methods=["GET"]),
            Route("/auth/setup", auth_setup, methods=["POST"]),
            Route("/auth/login", auth_login, methods=["POST"]),
            Route("/auth/logout", auth_logout, methods=["POST"]),
            Route("/auth/change-password", auth_change_password, methods=["POST"]),
            Route("/api/login", auth_login, methods=["POST"]),
            Route("/api/logout", auth_logout, methods=["POST"]),
            Route("/api/session_status", api_session_status, methods=["GET"]),
            Route("/api/password", api_password_put, methods=["PUT"]),
            # API
            Route("/api/buckets", api_buckets, methods=["GET"]),
            Route("/api/bucket/{bucket_id}", api_bucket_detail, methods=["GET"]),
            Route("/api/bucket/{bucket_id}", api_bucket_update, methods=["PATCH"]),
            Route("/api/bucket/{bucket_id}", api_bucket_delete, methods=["DELETE"]),
            Route("/api/sessions", api_sessions, methods=["GET"]),
            Route("/api/stats", api_stats, methods=["GET"]),
            Route("/api/search", api_search, methods=["GET"]),
            Route("/api/memories", api_memories, methods=["GET"]),
            Route("/api/memories", api_memory_create, methods=["POST"]),
            Route("/api/memories/{bucket_id}", api_memory_update, methods=["PUT"]),
            Route("/api/memories/{bucket_id}", api_memory_delete, methods=["DELETE"]),
            Route("/api/analyze", api_analyze, methods=["POST"]),
            Route("/api/grow", api_grow, methods=["POST"]),
            Route("/api/status", api_status_compat, methods=["GET"]),
            Route("/api/config", api_config_get, methods=["GET"]),
            Route("/api/config", api_config_put, methods=["PUT"]),
            Route("/api/backfill_embeddings", api_backfill_embeddings, methods=["POST"]),
            Route("/api/export/config", api_export_config, methods=["GET"]),
            Route("/api/export/memories", api_export_memories, methods=["GET"]),
            Route("/api/import/config", api_import_config, methods=["POST"]),
            Route("/api/import/memories", api_import_memories, methods=["POST"]),
            Route("/api/backups/list", api_backups_list, methods=["GET"]),
            Route("/api/backups/delete", api_backups_delete, methods=["POST"]),
        ]

        app = Starlette(routes=routes)
        return app

    async def start(self, host: str = "127.0.0.1", port: int = 2140) -> None:
        """Start the dashboard server in a background task."""
        try:
            import uvicorn

            app = self.build_app()
            config = uvicorn.Config(
                app, host=host, port=port, log_level="warning", access_log=False
            )
            self._server = uvicorn.Server(config)
            self._task = asyncio.create_task(self._server.serve())
            logger.info(f"[memory] Dashboard: http://{host}:{port}/")
        except OSError as e:
            logger.error(f"[memory] Dashboard 启动失败 (端口 {port} 被占用?): {e}")
        except Exception as e:
            logger.error(f"[memory] Dashboard 启动失败: {e}")

    async def stop(self) -> None:
        """Stop the dashboard server."""
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            finally:
                self._task = None
                self._server = None
