"""SQLite schema definitions and migration runner.

The plugin uses a single ``memory.db`` file with two domain tables
(``memories``, ``embeddings``) plus a bookkeeping ``schema_version`` table.

Migrations are applied in numeric order; once applied a row is recorded so
re-running ``apply_migrations`` is idempotent. Future schema changes
should be appended as ``MIGRATIONS[n] = (description, sql_or_callable)``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from .db import Database

logger = logging.getLogger("astrbot_plugin_ob_memory.schema")


# ---------------------------------------------------------------------------
# Initial schema (version 1).
# ``memories.session_id`` carries the AstrBot ``unified_msg_origin`` which
# is the cornerstone of session isolation. Every query in the manager joins
# / filters on this column.
# ---------------------------------------------------------------------------
SCHEMA_V1: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS memories (
        id               TEXT PRIMARY KEY,
        session_id       TEXT NOT NULL,
        name             TEXT NOT NULL DEFAULT '',
        content          TEXT NOT NULL,
        domain           TEXT NOT NULL DEFAULT '[]',
        tags             TEXT NOT NULL DEFAULT '[]',
        valence          REAL NOT NULL DEFAULT 0.5,
        arousal          REAL NOT NULL DEFAULT 0.3,
        importance       INTEGER NOT NULL DEFAULT 5,
        bucket_type      TEXT NOT NULL DEFAULT 'dynamic',
        pinned           INTEGER NOT NULL DEFAULT 0,
        resolved         INTEGER NOT NULL DEFAULT 0,
        digested         INTEGER NOT NULL DEFAULT 0,
        model_valence    REAL,
        source_bucket_id TEXT,
        activation_count REAL NOT NULL DEFAULT 0.0,
        created_at       REAL NOT NULL,
        last_active_at   REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_mem_session_type   ON memories(session_id, bucket_type)",
    "CREATE INDEX IF NOT EXISTS idx_mem_session_active ON memories(session_id, last_active_at)",
    "CREATE INDEX IF NOT EXISTS idx_mem_session_pinned ON memories(session_id, pinned)",
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        bucket_id  TEXT PRIMARY KEY,
        vector     BLOB NOT NULL,
        dim        INTEGER NOT NULL,
        updated_at REAL NOT NULL,
        FOREIGN KEY (bucket_id) REFERENCES memories(id) ON DELETE CASCADE
    )
    """,
)


SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL
)
"""


# A migration is either a tuple of SQL strings or an async callable that
# receives the ``Database`` instance and applies any custom logic.
Migration = tuple[str, ...] | Callable[[Database], Awaitable[None]]


MIGRATIONS: dict[int, tuple[str, Migration]] = {
    1: ("initial schema: memories + embeddings", SCHEMA_V1),
}


SCHEMA_VERSION: int = max(MIGRATIONS.keys())
"""The latest schema version known to this build of the plugin."""


async def _current_version(db: Database) -> int:
    """Return the highest applied schema version, or 0 if untracked."""
    await db.execute(SCHEMA_VERSION_TABLE)
    row = await db.fetch_one("SELECT MAX(version) AS v FROM schema_version")
    if row is None or row["v"] is None:
        return 0
    return int(row["v"])


async def _apply_one(db: Database, version: int, migration: Migration) -> None:
    """Apply a single migration and record it in ``schema_version``."""
    if callable(migration):
        await migration(db)
    else:
        for stmt in migration:
            await db.execute(stmt)
    await db.execute(
        "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
        (version, time.time()),
    )


async def apply_migrations(db: Database) -> None:
    """Bring the database up to ``SCHEMA_VERSION``.

    Safe to call on every startup: already-applied versions are skipped via
    the ``schema_version`` table. Each migration runs inside its own
    transaction so a failure mid-migration does not leave a half-applied
    schema (subject to SQLite DDL transactionality, which is real for
    ``CREATE TABLE`` / ``CREATE INDEX``).
    """
    current = await _current_version(db)
    pending = sorted(v for v in MIGRATIONS if v > current)
    if not pending:
        logger.debug(f"schema already at version {current}")
        return

    for version in pending:
        description, migration = MIGRATIONS[version]
        logger.info(f"applying schema migration {version}: {description}")
        async with db.transaction():
            await _apply_one(db, version, migration)
