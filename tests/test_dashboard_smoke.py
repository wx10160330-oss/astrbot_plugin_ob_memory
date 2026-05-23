"""Smoke tests for the dashboard module.

Verifies the server can be built and basic auth flow works without an
actual HTTP server running.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import io
import json
import time
import zipfile

import pytest

from astrbot_plugin_ob_memory.dashboard.auth import AuthStore
from astrbot_plugin_ob_memory.dashboard.server import DashboardServer


def _make_export_zip(memories, embeddings=None):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"format": "astrbot-memory-dashboard-export-v1", "kind": "memories"}),
        )
        zf.writestr("memories.json", json.dumps({"memories": memories}))
        zf.writestr("embeddings.json", json.dumps({"embeddings": embeddings or []}))
    return payload.getvalue()


def _make_plugin_for_dashboard(tmp_path: Path):
    plugin = MagicMock()
    plugin.config = {
        "digest_model": "deepseek-chat",
        "digest_prompt": "digest prompt",
        "embedding_provider_id": "emb-1",
        "tagging_enabled": True,
        "merge_threshold": 0.9,
    }
    plugin.decay = None
    plugin.search = None
    plugin.tagger = None
    plugin.writer = MagicMock()
    plugin.writer.digest_prompt = "digest prompt"
    plugin.writer.tagging_enabled = True
    plugin.writer.merge_enabled = True
    plugin.writer.merge_threshold = 0.9
    bucket = SimpleNamespace(
        id="bucket-1",
        session_id="session-A",
        name="初始名",
        content="拿到 offer",
        domain=["工作"],
        tags=["offer"],
        valence=0.8,
        arousal=0.4,
        importance=7,
        bucket_type="dynamic",
        pinned=False,
        resolved=False,
        digested=False,
        activation_count=0.0,
        created_at=time.time(),
        last_active_at=time.time(),
        model_valence=None,
        source_bucket_id=None,
    )
    buckets = {bucket.id: bucket}
    plugin.manager = MagicMock()

    async def fake_list_sessions():
        return ["session-A"]

    async def fake_list_by_session(session_id, include_archived=False):
        return list(buckets.values()) if session_id == "session-A" else []

    async def fake_get(session_id, bucket_id):
        if session_id == "session-A":
            return buckets.get(bucket_id)
        return None

    async def fake_create(new_bucket):
        buckets[new_bucket.id] = new_bucket
        return new_bucket.id

    plugin.manager.list_sessions = fake_list_sessions
    plugin.manager.list_by_session = fake_list_by_session
    plugin.manager.get = fake_get
    plugin.manager.create = fake_create

    plugin.embedding = MagicMock()
    stored_vectors = {"bucket-1": [0.1, 0.2, 0.3]}

    async def fake_embedding_get(bucket_id):
        return stored_vectors.get(bucket_id)

    async def fake_generate_and_store(bucket_id, content):
        stored_vectors[bucket_id] = [0.9, 0.8]
        return True

    plugin.embedding.enabled = True
    plugin.embedding.get = fake_embedding_get
    plugin.embedding.generate_and_store = fake_generate_and_store
    plugin.embedding.pack_vector = lambda vec: b"packed"
    plugin.db = MagicMock()

    class _Txn:
        async def __aenter__(self):
            self.conn = AsyncMock()
            return self.conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    plugin.db.transaction = lambda: _Txn()
    return plugin, buckets, stored_vectors


def test_auth_store_setup_and_login(tmp_path: Path):
    """Basic flow: setup → login → revoke."""
    store = AuthStore(tmp_path)
    assert store.setup_needed
    assert not store.env_locked

    # Setup
    assert store.setup_password("hello1234")
    assert not store.setup_needed

    # Cannot setup again
    assert not store.setup_password("anothertry")

    # Wrong password
    assert not store.check_password("wrong")
    assert store.check_password("hello1234")

    # Issue & validate session
    token = store.create_session()
    assert store.validate_session(token)

    # Revoke
    store.revoke_session(token)
    assert not store.validate_session(token)


def test_auth_store_change_password(tmp_path: Path):
    store = AuthStore(tmp_path)
    store.setup_password("oldpass")

    # Wrong current → fail
    assert not store.change_password("wrong", "newpass")
    assert store.check_password("oldpass")

    # Right current → success
    assert store.change_password("oldpass", "newpass")
    assert not store.check_password("oldpass")
    assert store.check_password("newpass")


def test_auth_store_min_length(tmp_path: Path):
    store = AuthStore(tmp_path)
    # Too short
    assert not store.setup_password("abc")
    assert store.setup_needed


def test_auth_store_persistence(tmp_path: Path):
    """Verify password survives restart."""
    store1 = AuthStore(tmp_path)
    store1.setup_password("persistent")

    store2 = AuthStore(tmp_path)
    assert not store2.setup_needed
    assert store2.check_password("persistent")


def test_auth_store_env_var_override(tmp_path: Path, monkeypatch):
    """When env var is set, it overrides file password."""
    monkeypatch.setenv("MEMORY_DASHBOARD_PASSWORD", "envpass")
    store = AuthStore(tmp_path)
    assert store.env_locked
    assert not store.setup_needed
    assert store.check_password("envpass")
    # Cannot change password when env-locked
    assert not store.change_password("envpass", "newpass")


def test_dashboard_server_builds_app(tmp_path: Path):
    """The Starlette app builds without errors and has the expected routes."""
    plugin = MagicMock()
    plugin.manager = None
    plugin.search = None
    plugin.decay = None
    plugin.embedding = None
    plugin.writer = None
    plugin.tagger = None
    plugin.config = {}

    server = DashboardServer(plugin, tmp_path)
    app = server.build_app()
    assert app is not None

    # Verify expected paths are registered
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/" in paths
    assert "/health" in paths
    assert "/dashboard" in paths
    assert "/manifest.json" in paths
    assert "/icons/{filename:path}" in paths
    assert "/auth/status" in paths
    assert "/auth/login" in paths
    assert "/auth/setup" in paths
    assert "/api/buckets" in paths
    assert "/api/bucket/{bucket_id}" in paths
    assert "/api/stats" in paths
    assert "/api/search" in paths
    assert "/api/memories" in paths
    assert "/api/memories/{bucket_id}" in paths
    assert "/api/analyze" in paths
    assert "/api/grow" in paths
    assert "/api/status" in paths
    assert "/api/config" in paths
    assert "/api/session_status" in paths
    assert "/api/password" in paths
    assert "/api/backfill_embeddings" in paths
    assert "/api/export/config" in paths
    assert "/api/export/memories" in paths
    assert "/api/import/config" in paths
    assert "/api/import/memories" in paths
    assert "/api/backups/list" in paths
    assert "/api/backups/delete" in paths


def test_nested_webui_config_takes_precedence():
    config = {
        "webui": {"enabled": True, "host": "0.0.0.0", "port": 2140},
        "enabled": True,
        "host": "127.0.0.1",
        "port": 9999,
    }

    webui_config = config.get("webui", {})
    if not isinstance(webui_config, dict):
        webui_config = {}
    enabled = bool(webui_config.get("enabled", config.get("enabled", True)))
    host = str(webui_config.get("host", config.get("host", "127.0.0.1")))
    port = int(webui_config.get("port", config.get("port", 2140)))

    assert enabled is True
    assert host == "0.0.0.0"
    assert port == 2140


def test_legacy_top_level_webui_config_still_works():
    config = {"enabled": True, "host": "0.0.0.0", "port": 2140}

    webui_config = config.get("webui", {})
    if not isinstance(webui_config, dict):
        webui_config = {}
    enabled = bool(webui_config.get("enabled", config.get("enabled", True)))
    host = str(webui_config.get("host", config.get("host", "127.0.0.1")))
    port = int(webui_config.get("port", config.get("port", 2140)))

    assert enabled is True
    assert host == "0.0.0.0"
    assert port == 2140


def test_dashboard_pulse_page_exposes_maintenance_actions():
    html = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "index.html"
    content = html.read_text(encoding="utf-8")

    assert "btn_export_config" in content
    assert "btn_export_memories" in content
    assert "btn_fix_emb" in content
    assert "triggerImportConfig()" in content
    assert "triggerImportMergeMemory()" in content
    assert "triggerRestoreReplace()" in content
    assert "showBackupsList()" in content
    assert "后续再接" not in content


@pytest.mark.asyncio
async def test_dashboard_health_endpoint(tmp_path: Path):
    """/health returns 200 without auth."""
    from starlette.testclient import TestClient

    plugin = MagicMock()
    server = DashboardServer(plugin, tmp_path)
    app = server.build_app()

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_dashboard_api_requires_auth(tmp_path: Path):
    """/api/* endpoints return 401 without a session cookie."""
    from starlette.testclient import TestClient

    plugin = MagicMock()
    server = DashboardServer(plugin, tmp_path)
    app = server.build_app()

    with TestClient(app) as client:
        resp = client.get("/api/buckets")
        assert resp.status_code == 401
        assert resp.json() == {"detail": "auth required"}


@pytest.mark.asyncio
async def test_dashboard_login_flow(tmp_path: Path):
    """End-to-end: setup → login → access protected endpoint."""
    from starlette.testclient import TestClient

    plugin = MagicMock()
    plugin.manager = MagicMock()

    async def fake_list_sessions():
        return []

    async def fake_count_in_session(sid):
        return {}

    plugin.manager.list_sessions = fake_list_sessions
    plugin.manager.count_in_session = fake_count_in_session
    plugin.decay = None
    plugin.embedding = None

    server = DashboardServer(plugin, tmp_path)
    app = server.build_app()

    with TestClient(app) as client:
        # Status: setup needed
        resp = client.get("/auth/status")
        assert resp.status_code == 200
        assert resp.json()["setup_needed"] is True

        # Setup
        resp = client.post("/auth/setup", json={"password": "test1234"})
        assert resp.status_code == 200
        assert resp.cookies.get("memory_dashboard_session")

        # Now /api/stats accessible (cookie carried automatically)
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data


@pytest.mark.asyncio
async def test_dashboard_memories_crud_compat(tmp_path: Path):
    from starlette.testclient import TestClient

    plugin = MagicMock()
    plugin.decay = None
    plugin.search = None
    plugin.embedding = None
    plugin.tagger = AsyncMock()
    plugin.tagger.analyze = AsyncMock(
        return_value={
            "domain": ["工作"],
            "valence": 0.8,
            "arousal": 0.4,
            "tags": ["offer"],
            "suggested_name": "offer",
            "importance": 7,
        }
    )
    plugin.tagger.digest = AsyncMock(
        return_value=[
            {
                "name": "offer",
                "content": "拿到 offer",
                "importance": 7,
                "tags": ["offer"],
                "domain": ["工作"],
                "valence": 0.8,
                "arousal": 0.4,
            }
        ]
    )
    plugin.writer = MagicMock()
    plugin.writer.digest_prompt = ""
    bucket = SimpleNamespace(
        id="bucket-1",
        session_id="session-A",
        name="初始名",
        content="拿到 offer",
        domain=["工作"],
        tags=["offer"],
        valence=0.8,
        arousal=0.4,
        importance=7,
        bucket_type="dynamic",
        pinned=False,
        resolved=False,
        digested=False,
        activation_count=0.0,
        created_at=time.time(),
        last_active_at=time.time(),
    )
    plugin.writer.hold = AsyncMock(
        return_value=SimpleNamespace(bucket_id="bucket-1", was_merged=False, target_bucket=bucket)
    )
    plugin.manager = MagicMock()
    plugin.config = {}

    async def fake_list_sessions():
        return ["session-A"]

    async def fake_list_by_session(session_id, include_archived=False):
        return [bucket] if session_id == "session-A" else []

    async def fake_get(session_id, bucket_id):
        if session_id == "session-A" and bucket_id == "bucket-1":
            return bucket
        return None

    async def fake_update(session_id, bucket_id, **fields):
        for key, value in fields.items():
            setattr(bucket, key, value)
        return bucket

    async def fake_delete(session_id, bucket_id):
        return session_id == "session-A" and bucket_id == "bucket-1"

    plugin.manager.list_sessions = fake_list_sessions
    plugin.manager.list_by_session = fake_list_by_session
    plugin.manager.get = fake_get
    plugin.manager.update = fake_update
    plugin.manager.delete = fake_delete

    server = DashboardServer(plugin, tmp_path)
    app = server.build_app()

    with TestClient(app) as client:
        assert client.post("/auth/setup", json={"password": "test1234"}).status_code == 200

        resp = client.get("/api/memories")
        assert resp.status_code == 200
        assert resp.json()[0]["id"] == "bucket-1"

        resp = client.post("/api/memories", json={"content": "拿到 offer", "name": "新记忆"})
        assert resp.status_code == 201
        plugin.writer.hold.assert_awaited()

        resp = client.put("/api/memories/bucket-1", json={"name": "已修改", "tags": ["offer", "job"]})
        assert resp.status_code == 200
        assert resp.json()["name"] == "已修改"
        assert resp.json()["tags"] == ["offer", "job"]

        resp = client.post("/api/analyze", json={"content": "拿到 offer"})
        assert resp.status_code == 200
        assert resp.json()["domain"] == ["工作"]
        assert resp.json()["importance"] == 7

        plugin.writer.hold.reset_mock()
        plugin.writer._persist_with_analysis = AsyncMock(
            return_value=SimpleNamespace(target_bucket=bucket)
        )
        resp = client.post("/api/memories", json={
            "content": "拿到 offer",
            "name": "",
            "importance": resp.json()["importance"],
            "tags": resp.json()["tags"],
            "domain": resp.json()["domain"],
            "valence": resp.json()["valence"],
            "arousal": resp.json()["arousal"],
            "analysis": resp.json(),
        })
        assert resp.status_code == 201
        plugin.writer._persist_with_analysis.assert_awaited()
        plugin.writer.hold.assert_not_called()
        _args, kwargs = plugin.writer._persist_with_analysis.await_args
        assert kwargs["analysis"]["importance"] == 7
        assert kwargs["tags"] == ["offer"]
        assert kwargs["valence"] == 0.8
        assert kwargs["arousal"] == 0.4

        resp = client.post("/api/grow", json={"content": "今天拿到 offer，还很开心"})
        assert resp.status_code == 200
        assert resp.json()[0]["name"] == "offer"

        resp = client.delete("/api/memories/bucket-1")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


@pytest.mark.asyncio
async def test_dashboard_compat_export_import_and_backups(tmp_path: Path):
    from starlette.testclient import TestClient

    plugin, buckets, stored_vectors = _make_plugin_for_dashboard(tmp_path)

    server = DashboardServer(plugin, tmp_path)
    app = server.build_app()

    with TestClient(app) as client:
        assert client.post("/auth/setup", json={"password": "test1234"}).status_code == 200

        config_resp = client.get("/api/export/config")
        assert config_resp.status_code == 200
        assert config_resp.json()["runtime_only"] is True

        memories_resp = client.get("/api/export/memories")
        assert memories_resp.status_code == 200
        assert memories_resp.headers["content-type"].startswith("application/zip")

        dry_run_resp = client.post("/api/backfill_embeddings", json={"dry_run": True})
        assert dry_run_resp.status_code == 200
        assert dry_run_resp.json()["missing"] == 0

        stored_vectors.clear()
        exec_resp = client.post("/api/backfill_embeddings", json={})
        assert exec_resp.status_code == 200
        assert exec_resp.json()["success"] == 1
        assert stored_vectors["bucket-1"] == [0.9, 0.8]

        import_config = io.BytesIO(
            json.dumps(
                {
                    "format": "astrbot-memory-dashboard-export-v1",
                    "config": {
                        "dehydration": {"model": "m2", "prompt": "p2"},
                        "embedding": {"provider_id": "emb-2"},
                        "tagging_enabled": False,
                        "merge_threshold": 0.5,
                    },
                }
            ).encode("utf-8")
        )
        import_cfg_resp = client.post(
            "/api/import/config",
            files={"file": ("config.json", import_config, "application/json")},
        )
        assert import_cfg_resp.status_code == 200
        assert plugin.config["digest_model"] == "m2"
        assert plugin.writer.merge_enabled is False

        import_zip = _make_export_zip(
            [
                {
                    "id": "bucket-2",
                    "session_id": "session-A",
                    "name": "新记忆",
                    "content": "新的内容",
                    "domain": ["工作"],
                    "tags": ["new"],
                    "valence": 0.6,
                    "arousal": 0.4,
                    "importance": 6,
                    "bucket_type": "dynamic",
                    "pinned": False,
                    "resolved": False,
                    "digested": False,
                    "activation_count": 0.0,
                    "created_at": 2.0,
                    "last_active_at": 2.0,
                }
            ]
        )

        import_dry_run = client.post(
            "/api/import/memories",
            data={"mode": "dry_run"},
            files={"file": ("memories.zip", io.BytesIO(import_zip), "application/zip")},
        )
        assert import_dry_run.status_code == 200
        assert import_dry_run.json()["new_count"] == 1

        import_merge = client.post(
            "/api/import/memories",
            data={"mode": "merge"},
            files={"file": ("memories.zip", io.BytesIO(import_zip), "application/zip")},
        )
        assert import_merge.status_code == 200
        assert import_merge.json()["added"] == 1
        assert "bucket-2" in buckets

        backups_resp = client.get("/api/backups/list")
        assert backups_resp.status_code == 200
        backup_name = backups_resp.json()["items"][0]["name"]



@pytest.mark.asyncio
async def test_dashboard_password_api_accepts_ui_field_names(tmp_path: Path):
    from starlette.testclient import TestClient

    plugin, _buckets, _stored_vectors = _make_plugin_for_dashboard(tmp_path)
    server = DashboardServer(plugin, tmp_path)
    app = server.build_app()

    with TestClient(app) as client:
        assert client.post("/auth/setup", json={"password": "test1234"}).status_code == 200

        resp = client.put(
            "/api/password",
            json={"current": "test1234", "new": "newpass5678"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        client.post("/api/logout")
        relogin = client.post("/api/login", json={"password": "newpass5678"})
        assert relogin.status_code == 200


@pytest.mark.asyncio
async def test_dashboard_config_round_trip_includes_ui_fields(tmp_path: Path):
    from starlette.testclient import TestClient

    plugin, _buckets, _stored_vectors = _make_plugin_for_dashboard(tmp_path)
    server = DashboardServer(plugin, tmp_path)
    app = server.build_app()

    with TestClient(app) as client:
        assert client.post("/auth/setup", json={"password": "test1234"}).status_code == 200

        get_resp = client.get("/api/config")
        assert get_resp.status_code == 200
        payload = get_resp.json()
        assert payload["dehydration"]["model"] == "deepseek-chat"
        assert payload["dehydration"]["prompt"] == "digest prompt"
        assert payload["dehydration"]["api_key"] == ""
        assert payload["dehydration"]["base_url"] == ""
        assert payload["embedding"]["provider_id"] == "emb-1"
        assert payload["embedding"]["api_key"] == ""
        assert payload["embedding"]["base_url"] == ""
        assert payload["embedding"]["model"] == ""

        put_resp = client.put(
            "/api/config",
            json={
                "dehydration": {
                    "api_key": "sk-dehy",
                    "base_url": "https://dehy.example/v1",
                    "model": "deepseek-v2",
                    "prompt": "new prompt",
                },
                "embedding": {
                    "provider_id": "emb-2",
                    "api_key": "sk-emb",
                    "base_url": "https://emb.example/v1",
                    "model": "bge-m3",
                },
                "tagging_enabled": False,
                "merge_threshold": 0.5,
            },
        )
        assert put_resp.status_code == 200
        assert plugin.config["digest_model"] == "deepseek-v2"
        assert plugin.config["digest_prompt"] == "new prompt"
        assert plugin.config["embedding_provider_id"] == "emb-2"
        assert plugin.config["tagging_enabled"] is False
        assert plugin.config["merge_threshold"] == 0.5
        assert plugin.config["dashboard_dehydration_api_key"] == "sk-dehy"
        assert plugin.config["dashboard_dehydration_base_url"] == "https://dehy.example/v1"
        assert plugin.config["dashboard_embedding_api_key"] == "sk-emb"
        assert plugin.config["dashboard_embedding_base_url"] == "https://emb.example/v1"
        assert plugin.config["dashboard_embedding_model"] == "bge-m3"


@pytest.mark.asyncio
async def test_dashboard_import_memories_rejects_invalid_backup_name(tmp_path: Path):
    from starlette.testclient import TestClient

    plugin = MagicMock()
    plugin.manager = None
    plugin.search = None
    plugin.decay = None
    plugin.embedding = None
    plugin.writer = None
    plugin.tagger = None
    plugin.config = {}

    server = DashboardServer(plugin, tmp_path)
    app = server.build_app()

    with TestClient(app) as client:
        assert client.post("/auth/setup", json={"password": "test1234"}).status_code == 200
        resp = client.post("/api/backups/delete", json={"name": "../evil.zip"})
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_dashboard_manifest_and_icon_routes(tmp_path: Path):
    from starlette.testclient import TestClient

    plugin = MagicMock()
    plugin.manager = None
    plugin.search = None
    plugin.decay = None
    plugin.embedding = None
    plugin.writer = None
    plugin.tagger = None
    plugin.config = {}

    server = DashboardServer(plugin, tmp_path)
    app = server.build_app()

    with TestClient(app) as client:
        manifest_resp = client.get("/manifest.json")
        assert manifest_resp.status_code == 200
        manifest = manifest_resp.json()
        icons = manifest["icons"]
        assert icons[0]["src"] == "/static/icon-192.png"
        assert icons[1]["src"] == "/static/icon-512.png"
        light_icons = manifest["light_icons"]
        assert light_icons[0]["src"] == "/static/icon-192.png"
        assert light_icons[1]["src"] == "/static/icon-512.png"
        dark_icons = manifest["dark_icons"]
        assert dark_icons[0]["src"] == "/static/icon-dark-192.png"
        assert dark_icons[1]["src"] == "/static/icon-dark-512.png"

        icon_resp = client.get("/icons/icon-192.png")
        assert icon_resp.status_code == 200
        assert icon_resp.headers["content-type"].startswith("image/png")
