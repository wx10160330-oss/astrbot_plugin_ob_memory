"""Smoke tests for the storage layer.

Verifies that:
- A fresh database can be opened and closed.
- Migrations bring an empty file to the latest version.
- Re-applying migrations on an already-up-to-date file is a no-op.
- ``ON DELETE CASCADE`` from ``memories`` to ``embeddings`` works.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from astrbot_plugin_ob_memory.storage import Database, SCHEMA_VERSION, apply_migrations


pytestmark = pytest.mark.asyncio


async def _open_fresh_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    return db


async def test_fresh_database_reaches_latest_schema(tmp_path: Path):
    db = await _open_fresh_db(tmp_path)
    try:
        assert db.is_connected
        row = await db.fetch_one(
            "SELECT MAX(version) AS v FROM schema_version"
        )
        assert row is not None
        assert row["v"] == SCHEMA_VERSION
    finally:
        await db.close()


async def test_migrations_are_idempotent(tmp_path: Path):
    db = await _open_fresh_db(tmp_path)
    try:
        # Second invocation must not raise and must not duplicate rows.
        await apply_migrations(db)
        rows = await db.fetch_all("SELECT version FROM schema_version")
        versions = sorted(r["version"] for r in rows)
        assert versions == sorted(set(versions))
        assert versions[-1] == SCHEMA_VERSION
    finally:
        await db.close()


async def test_transaction_rolls_back_all_statements(tmp_path: Path):
    db = await _open_fresh_db(tmp_path)
    try:
        now = time.time()
        with pytest.raises(RuntimeError):
            async with db.transaction():
                await db.execute(
                    "INSERT INTO memories (id, session_id, content, created_at, last_active_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("tx1", "session-A", "hello", now, now),
                )
                raise RuntimeError("boom")

        row = await db.fetch_one("SELECT id FROM memories WHERE id = ?", ("tx1",))
        assert row is None
    finally:
        await db.close()


async def test_execute_commits_outside_transaction(tmp_path: Path):
    db = await _open_fresh_db(tmp_path)
    try:
        now = time.time()
        await db.execute(
            "INSERT INTO memories (id, session_id, content, created_at, last_active_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("tx2", "session-A", "hello", now, now),
        )

        row = await db.fetch_one("SELECT id FROM memories WHERE id = ?", ("tx2",))
        assert row is not None
    finally:
        await db.close()


async def test_embeddings_cascade_on_memory_delete(tmp_path: Path):
    db = await _open_fresh_db(tmp_path)
    try:
        now = time.time()
        await db.execute(
            "INSERT INTO memories (id, session_id, content, created_at, last_active_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("abc123", "session-A", "hello", now, now),
        )
        await db.execute(
            "INSERT INTO embeddings (bucket_id, vector, dim, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("abc123", b"\x00" * 8, 2, now),
        )

        rows = await db.fetch_all("SELECT bucket_id FROM embeddings")
        assert [r["bucket_id"] for r in rows] == ["abc123"]

        await db.execute("DELETE FROM memories WHERE id = ?", ("abc123",))
        rows = await db.fetch_all("SELECT bucket_id FROM embeddings")
        assert rows == []
    finally:
        await db.close()


async def test_session_isolation_via_indexes(tmp_path: Path):
    """Inserts buckets across two sessions and verifies basic SQL filter.

    This is a pre-test for Phase 2's MemoryManager — we just exercise the
    raw column filter so the schema is proven before higher layers depend
    on it.
    """
    db = await _open_fresh_db(tmp_path)
    try:
        now = time.time()
        rows = [
            ("a1", "session-A", "content A1", now, now),
            ("a2", "session-A", "content A2", now, now),
            ("b1", "session-B", "content B1", now, now),
        ]
        for r in rows:
            await db.execute(
                "INSERT INTO memories (id, session_id, content, created_at, last_active_at) "
                "VALUES (?, ?, ?, ?, ?)",
                r,
            )

        a_rows = await db.fetch_all(
            "SELECT id FROM memories WHERE session_id = ?", ("session-A",)
        )
        b_rows = await db.fetch_all(
            "SELECT id FROM memories WHERE session_id = ?", ("session-B",)
        )

        assert sorted(r["id"] for r in a_rows) == ["a1", "a2"]
        assert [r["id"] for r in b_rows] == ["b1"]
    finally:
        await db.close()
