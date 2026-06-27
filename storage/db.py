"""Async-friendly wrapper around aiosqlite for the memory plugin.

A single connection is held for the lifetime of the plugin. SQLite's
threading model with WAL journal mode makes this safe for the AstrBot
use case (modest read/write rates, single Python process). Every query
goes through this class so we have one place to enforce defaults like
``foreign_keys=ON`` and JSON1 availability checks.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger("astrbot_plugin_ob_memory.db")


class Database:
    """Lightweight async SQLite wrapper used by the memory plugin.

    Methods are intentionally narrow — the manager layer composes them
    rather than depending on the full aiosqlite surface area. This keeps
    swapping the storage backend (e.g. to an in-memory mock during tests)
    a one-class change.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """Open the connection and apply baseline pragmas.

        Pragmas:
        - ``journal_mode=WAL`` — concurrent reads while a writer holds the
          lock, mandatory for a long-lived plugin database.
        - ``foreign_keys=ON`` — enables the ``embeddings → memories``
          ON DELETE CASCADE we rely on.
        - ``synchronous=NORMAL`` — good throughput/safety tradeoff for WAL.
        """
        if self._conn is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.path), isolation_level=None)
        # Make rows behave like dicts (row["col"]).
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.commit()
        logger.debug(f"connected to {self.path}")

    async def close(self) -> None:
        """Close the connection and release the WAL files."""
        if self._conn is None:
            return
        try:
            await self._conn.close()
        finally:
            self._conn = None

    @property
    def is_connected(self) -> bool:
        """``True`` once ``connect`` has succeeded and ``close`` not run."""
        return self._conn is not None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected. Call connect() first.")
        return self._conn

    def _should_autocommit(self, sql: str) -> bool:
        conn = self._require_conn()
        if conn.in_transaction:
            return False
        return sql.lstrip()[:6].upper() in (
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
        )

    async def execute(
        self, sql: str, params: Iterable[Any] | None = None
    ) -> aiosqlite.Cursor:
        """Run a single statement; commits immediately for standalone writes.

        For batched mutations use ``transaction()``. When a caller has
        entered an explicit transaction block, this helper must not commit
        early or it would break rollback guarantees.
        """
        conn = self._require_conn()
        cursor = await conn.execute(sql, tuple(params or ()))
        if self._should_autocommit(sql):
            await conn.commit()
        return cursor

    async def executemany(
        self, sql: str, seq_of_params: Iterable[Iterable[Any]]
    ) -> None:
        """Bulk-mutation helper. Always commits at the end."""
        conn = self._require_conn()
        await conn.executemany(sql, [tuple(p) for p in seq_of_params])
        await conn.commit()

    async def fetch_one(
        self, sql: str, params: Iterable[Any] | None = None
    ) -> aiosqlite.Row | None:
        """Run ``sql`` and return the first row, or ``None``."""
        conn = self._require_conn()
        async with conn.execute(sql, tuple(params or ())) as cursor:
            return await cursor.fetchone()

    async def fetch_all(
        self, sql: str, params: Iterable[Any] | None = None
    ) -> list[aiosqlite.Row]:
        """Run ``sql`` and return all rows."""
        conn = self._require_conn()
        async with conn.execute(sql, tuple(params or ())) as cursor:
            return list(await cursor.fetchall())

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Run a block of statements inside a single transaction.

        SQLite transactions in aiosqlite are implicit — we explicitly
        ``BEGIN`` so the semantics are predictable, and commit on success
        / rollback on exception.
        """
        conn = self._require_conn()
        await conn.execute("BEGIN")
        try:
            yield conn
        except BaseException:
            await conn.rollback()
            raise
        else:
            await conn.commit()
