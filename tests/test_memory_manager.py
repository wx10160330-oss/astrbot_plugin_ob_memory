"""End-to-end CRUD + touch + ripple tests against a real SQLite file.

These tests use the temporary directory pytest provides for a fresh DB
per case. They cover Tasks 6 (CRUD with session scoping), 7 (touch
preserves identity), and the time ripple introduced for parity with
Ombre Brain B-03.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.core.models import MemoryBucket, new_bucket
from astrbot_plugin_ob_memory.storage import Database, apply_migrations

pytestmark = pytest.mark.asyncio


async def _open_manager(tmp_path: Path) -> tuple[Database, MemoryManager]:
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    return db, MemoryManager(db)


# ---------------------------------------------------------------------------
# CREATE / GET
# ---------------------------------------------------------------------------
async def test_create_then_get_round_trip(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        bucket = await mgr.create_simple(
            "session-A", "拿到了实习offer",
            name="实习offer", importance=7,
            valence=0.8, arousal=0.7,
            tags=["offer", "实习"], domain=["成长"],
        )
        loaded = await mgr.get("session-A", bucket.id)
        assert loaded is not None
        assert loaded.content == "拿到了实习offer"
        assert loaded.importance == 7
        assert loaded.tags == ["offer", "实习"]
    finally:
        await db.close()


async def test_create_clamps_out_of_range_inputs(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        bucket = MemoryBucket(
            id="abc123abc123",
            session_id="session-A",
            content="boom",
            valence=2.5,
            arousal=-1.0,
            importance=99,
            created_at=time.time(),
            last_active_at=time.time(),
        )
        await mgr.create(bucket)
        loaded = await mgr.get("session-A", "abc123abc123")
        assert loaded is not None
        assert loaded.valence == 1.0
        assert loaded.arousal == 0.0
        assert loaded.importance == 10
    finally:
        await db.close()


async def test_create_rejects_missing_session_or_id(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        b = new_bucket("session-A", "x")
        b.session_id = ""
        with pytest.raises(ValueError):
            await mgr.create(b)

        b = new_bucket("session-A", "x")
        b.id = ""
        with pytest.raises(ValueError):
            await mgr.create(b)
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# SESSION ISOLATION
# ---------------------------------------------------------------------------
async def test_session_isolation_get(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        b = await mgr.create_simple("session-A", "private memory")
        # Querying the bucket id from session B must return None — even
        # though the row exists, it must be invisible across sessions.
        assert await mgr.get("session-B", b.id) is None
        assert await mgr.get("session-A", b.id) is not None
    finally:
        await db.close()


async def test_session_isolation_list(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        await mgr.create_simple("session-A", "A1")
        await mgr.create_simple("session-A", "A2")
        await mgr.create_simple("session-B", "B1")

        a_list = await mgr.list_by_session("session-A")
        b_list = await mgr.list_by_session("session-B")
        assert {b.content for b in a_list} == {"A1", "A2"}
        assert {b.content for b in b_list} == {"B1"}
    finally:
        await db.close()


async def test_session_isolation_concurrent_writes(tmp_path: Path):
    """Spawn parallel writes against two sessions and verify zero leakage."""
    db, mgr = await _open_manager(tmp_path)
    try:
        async def write_many(session_id: str, prefix: str) -> None:
            for i in range(20):
                await mgr.create_simple(session_id, f"{prefix}-{i}")

        await asyncio.gather(
            write_many("session-A", "alpha"),
            write_many("session-B", "beta"),
        )

        a_list = await mgr.list_by_session("session-A")
        b_list = await mgr.list_by_session("session-B")
        assert all(b.content.startswith("alpha-") for b in a_list)
        assert all(b.content.startswith("beta-") for b in b_list)
        assert len(a_list) == 20
        assert len(b_list) == 20
    finally:
        await db.close()


async def test_list_sessions_returns_distinct(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        await mgr.create_simple("s1", "x")
        await mgr.create_simple("s1", "y")
        await mgr.create_simple("s2", "z")
        sessions = await mgr.list_sessions()
        assert sessions == ["s1", "s2"]
    finally:
        await db.close()


async def test_count_in_session(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        await mgr.create_simple("s", "x", bucket_type="dynamic")
        await mgr.create_simple("s", "p", pinned=True)  # → permanent
        await mgr.create_simple("s", "f", bucket_type="feel")
        counts = await mgr.count_in_session("s")
        assert counts["dynamic"] == 1
        assert counts["permanent"] == 1
        assert counts["feel"] == 1
        assert counts["archived"] == 0
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------------
async def test_update_partial_fields(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        b = await mgr.create_simple("s", "before", importance=5)
        updated = await mgr.update("s", b.id, content="after", importance=8)
        assert updated is not None
        assert updated.content == "after"
        assert updated.importance == 8

        # Re-read from DB to confirm persistence
        re_loaded = await mgr.get("s", b.id)
        assert re_loaded is not None
        assert re_loaded.content == "after"
        assert re_loaded.importance == 8
    finally:
        await db.close()


async def test_update_clamps_out_of_range(tmp_path: Path):
    """Property 11: edits via the manager must apply the same clamping."""
    db, mgr = await _open_manager(tmp_path)
    try:
        b = await mgr.create_simple("s", "x")
        updated = await mgr.update("s", b.id, valence=1.5, arousal=-0.4, importance=99)
        assert updated is not None
        assert updated.valence == 1.0
        assert updated.arousal == 0.0
        assert updated.importance == 10
    finally:
        await db.close()


async def test_update_pinned_invariant(tmp_path: Path):
    """Pinning must lock importance=10 and bucket_type=permanent."""
    db, mgr = await _open_manager(tmp_path)
    try:
        b = await mgr.create_simple("s", "x", importance=4)
        updated = await mgr.update("s", b.id, pinned=True)
        assert updated is not None
        assert updated.pinned is True
        assert updated.importance == 10
        assert updated.bucket_type == "permanent"
    finally:
        await db.close()


async def test_update_ignores_unknown_field(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        b = await mgr.create_simple("s", "x")
        updated = await mgr.update("s", b.id, completely_made_up=42, importance=6)
        assert updated is not None
        assert updated.importance == 6
    finally:
        await db.close()


async def test_update_returns_none_for_missing(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        result = await mgr.update("s", "no-such-id", importance=5)
        assert result is None
    finally:
        await db.close()


async def test_update_can_backdate_created_at(tmp_path: Path):
    """Dashboard / command callers must be able to override ``created_at``
    on manually inscribed memories (e.g. backdating an old event)."""
    db, mgr = await _open_manager(tmp_path)
    try:
        b = await mgr.create_simple("session-A", "old memory")
        backdated = b.created_at - 60 * 86400  # 60 days ago
        updated = await mgr.update(
            "session-A", b.id, created_at=backdated, last_active_at=backdated
        )
        assert updated is not None
        assert updated.created_at == pytest.approx(backdated)
        assert updated.last_active_at == pytest.approx(backdated)

        re_loaded = await mgr.get("session-A", b.id)
        assert re_loaded is not None
        assert re_loaded.created_at == pytest.approx(backdated)
    finally:
        await db.close()


async def test_update_isolated_per_session(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        b = await mgr.create_simple("session-A", "secret")
        # Trying to update from a different session must not affect row.
        result = await mgr.update("session-B", b.id, content="hijacked")
        assert result is None
        re_loaded = await mgr.get("session-A", b.id)
        assert re_loaded is not None
        assert re_loaded.content == "secret"
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# DELETE / ARCHIVE
# ---------------------------------------------------------------------------
async def test_delete_removes_bucket_and_embedding(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        b = await mgr.create_simple("s", "x")
        # Manually insert an embedding row for this bucket.
        await db.execute(
            "INSERT INTO embeddings (bucket_id, vector, dim, updated_at) VALUES (?, ?, ?, ?)",
            (b.id, b"\x00" * 8, 2, time.time()),
        )
        deleted = await mgr.delete("s", b.id)
        assert deleted is True

        assert await mgr.get("s", b.id) is None
        rows = await db.fetch_all(
            "SELECT bucket_id FROM embeddings WHERE bucket_id = ?", (b.id,)
        )
        assert rows == []
    finally:
        await db.close()


async def test_delete_isolated_per_session(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        b = await mgr.create_simple("session-A", "x")
        deleted = await mgr.delete("session-B", b.id)
        assert deleted is False
        assert await mgr.get("session-A", b.id) is not None
    finally:
        await db.close()


async def test_archive_changes_type(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        b = await mgr.create_simple("s", "x")
        ok = await mgr.archive("s", b.id)
        assert ok is True
        loaded = await mgr.get("s", b.id)
        assert loaded is not None
        assert loaded.bucket_type == "archived"

        # By default list_by_session excludes archived.
        active = await mgr.list_by_session("s")
        assert active == []

        # Explicitly include archived.
        all_ = await mgr.list_by_session("s", include_archived=True)
        assert len(all_) == 1
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# TOUCH + TIME RIPPLE  (Property 9 + Ombre Brain parity)
# ---------------------------------------------------------------------------
async def test_touch_only_updates_two_fields(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        b = await mgr.create_simple(
            "s", "x", importance=6, valence=0.4, arousal=0.6,
        )
        before_active = b.last_active_at
        before_count = b.activation_count

        # Sleep enough for last_active_at to advance even on a fast machine.
        await asyncio.sleep(0.02)
        await mgr.touch("s", b.id)

        after = await mgr.get("s", b.id)
        assert after is not None
        assert after.last_active_at > before_active
        assert after.activation_count == before_count + 1.0
        # Property 9: nothing else changed
        assert after.content == b.content
        assert after.name == b.name
        assert after.importance == b.importance
        assert after.valence == pytest.approx(b.valence)
        assert after.arousal == pytest.approx(b.arousal)
        assert after.bucket_type == b.bucket_type
        assert after.pinned == b.pinned
        assert after.resolved == b.resolved
        assert after.digested == b.digested
        assert after.created_at == pytest.approx(b.created_at)
    finally:
        await db.close()


async def test_touch_no_op_for_missing(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        # Should not raise even if the bucket doesn't exist.
        await mgr.touch("s", "missing-id")
    finally:
        await db.close()


async def test_time_ripple_boosts_neighbours(tmp_path: Path):
    """Buckets created within the time window get a +0.3 bump."""
    db, mgr = await _open_manager(tmp_path)
    try:
        now = time.time()

        # Create three siblings within a ±48h window of `source`.
        # We backdate created_at via a direct UPDATE because the public
        # API doesn't expose it (it shouldn't — only ripple may set it).
        async def insert_with_created_at(content: str, created: float) -> str:
            b = await mgr.create_simple("s", content)
            await db.execute(
                "UPDATE memories SET created_at = ?, last_active_at = ? "
                "WHERE id = ? AND session_id = ?",
                (created, created, b.id, "s"),
            )
            return b.id

        source_id = await insert_with_created_at("source", now)
        near_id = await insert_with_created_at("near", now - 6 * 3600)
        far_id = await insert_with_created_at("far", now - 7 * 86400)

        # Pinned bucket near in time — must be skipped.
        pinned = await mgr.create_simple("s", "core", pinned=True)
        await db.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (now - 1 * 3600, pinned.id),
        )

        # Trigger ripple. Use the raw method so we don't also trigger touch.
        await mgr.time_ripple("s", source_id, now)

        near = await mgr.get("s", near_id)
        far = await mgr.get("s", far_id)
        pin = await mgr.get("s", pinned.id)
        src = await mgr.get("s", source_id)

        assert near is not None and far is not None and pin is not None and src is not None
        assert near.activation_count == pytest.approx(0.3)
        assert far.activation_count == 0.0
        assert pin.activation_count == 0.0
        # Source bucket itself is excluded.
        assert src.activation_count == 0.0
    finally:
        await db.close()


async def test_time_ripple_caps_at_max(tmp_path: Path):
    """No more than ``TIME_RIPPLE_MAX_BUCKETS`` neighbours get bumped."""
    db, mgr = await _open_manager(tmp_path)
    try:
        now = time.time()
        ids: list[str] = []
        for i in range(8):
            b = await mgr.create_simple("s", f"n{i}")
            await db.execute(
                "UPDATE memories SET created_at = ? WHERE id = ?",
                (now - i * 3600, b.id),
            )
            ids.append(b.id)

        source_id = ids[0]
        await mgr.time_ripple("s", source_id, now)

        boosted = 0
        for bid in ids[1:]:
            b = await mgr.get("s", bid)
            assert b is not None
            if b.activation_count > 0:
                boosted += 1
        # Cap is 5 by default.
        assert boosted == 5
    finally:
        await db.close()


async def test_time_ripple_isolates_sessions(tmp_path: Path):
    """A touch in session-A must not bump buckets in session-B."""
    db, mgr = await _open_manager(tmp_path)
    try:
        now = time.time()
        a_source = await mgr.create_simple("session-A", "a-source")
        a_neighbour = await mgr.create_simple("session-A", "a-neighbour")
        b_neighbour = await mgr.create_simple("session-B", "b-neighbour")

        # Backdate everyone to within the window of `now`.
        for bid in (a_source.id, a_neighbour.id, b_neighbour.id):
            await db.execute(
                "UPDATE memories SET created_at = ? WHERE id = ?", (now, bid)
            )

        await mgr.time_ripple("session-A", a_source.id, now)

        a_n = await mgr.get("session-A", a_neighbour.id)
        b_n = await mgr.get("session-B", b_neighbour.id)
        assert a_n is not None and b_n is not None
        assert a_n.activation_count == pytest.approx(0.3)
        assert b_n.activation_count == 0.0
    finally:
        await db.close()


# ===========================================================================
# Persistent every_n_turns counter (schema v2 session_state)
# ===========================================================================
@pytest.mark.asyncio
async def test_auto_record_counter_defaults_to_zero(tmp_path: Path):
    """An unseen session reads back 0 — no row, no error."""
    db, mgr = await _open_manager(tmp_path)
    try:
        assert await mgr.get_auto_record_counter("fresh:session") == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_auto_record_counter_set_and_get(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        await mgr.set_auto_record_counter("qq:Group:1", 7)
        assert await mgr.get_auto_record_counter("qq:Group:1") == 7

        # Updating writes through (no duplicate row, ON CONFLICT UPDATE).
        await mgr.set_auto_record_counter("qq:Group:1", 12)
        assert await mgr.get_auto_record_counter("qq:Group:1") == 12

        # Sessions are isolated.
        assert await mgr.get_auto_record_counter("qq:Group:2") == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_auto_record_counter_bump(tmp_path: Path):
    db, mgr = await _open_manager(tmp_path)
    try:
        assert await mgr.bump_auto_record_counter("sid") == 1
        assert await mgr.bump_auto_record_counter("sid") == 2
        assert await mgr.bump_auto_record_counter("sid") == 3
        assert await mgr.get_auto_record_counter("sid") == 3
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_auto_record_counter_survives_reopen(tmp_path: Path):
    """The whole point of schema v2: closing + reopening the db must
    preserve the counter so a plugin / AstrBot restart doesn't zero out
    the user's accumulated turns."""
    db_path = tmp_path / "memory.db"

    db1 = Database(db_path)
    await db1.connect()
    await apply_migrations(db1)
    mgr1 = MemoryManager(db1)
    await mgr1.set_auto_record_counter("sid", 5)
    await mgr1.bump_auto_record_counter("sid")
    await mgr1.bump_auto_record_counter("sid")
    await db1.close()

    db2 = Database(db_path)
    await db2.connect()
    await apply_migrations(db2)
    mgr2 = MemoryManager(db2)
    try:
        assert await mgr2.get_auto_record_counter("sid") == 7
    finally:
        await db2.close()
