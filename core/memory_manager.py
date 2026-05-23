"""High-level CRUD layer over the memory database.

Every read and write goes through this module. Three rules are enforced
unconditionally:

1. **Session isolation.** Every query carries a ``session_id`` predicate;
   no method accepts ``session_id=None`` as "all sessions". Callers that
   genuinely want cross-session iteration use :meth:`list_sessions` and
   then iterate.
2. **Clamping.** All input passes through :func:`core.models.clamp_bucket`
   so ranges are guaranteed before the row hits SQLite.
3. **Embedding cleanup.** Deleting a bucket removes its embedding row in
   the same transaction (also enforced at the SQLite level by
   ``ON DELETE CASCADE``).

Higher-level features (auto-tagging, merging, the ``hold`` flow) live in
:class:`core.memory_writer.MemoryWriter`. The manager exposes the pure
CRUD surface plus :meth:`touch` / :meth:`time_ripple` which are needed
everywhere.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from typing import Any

from ..storage.db import Database
from .models import (
    VALID_BUCKET_TYPES,
    BucketType,
    MemoryBucket,
    clamp_bucket,
    new_bucket,
)

logger = logging.getLogger("astrbot_plugin_ob_memory.manager")


# ---------------------------------------------------------------------------
# Constants pulled out of class body so they are easy to override in tests.
# ---------------------------------------------------------------------------
TIME_RIPPLE_WINDOW_HOURS: float = 48.0
"""Buckets created within ±N hours of the touched bucket get a small boost."""

TIME_RIPPLE_INCREMENT: float = 0.3
"""How much activation_count is added to each rippled neighbour."""

TIME_RIPPLE_MAX_BUCKETS: int = 5
"""Cap on neighbours rippled per touch to bound I/O."""


# ---------------------------------------------------------------------------
# Row <-> bucket conversion
# ---------------------------------------------------------------------------
_COLUMNS: tuple[str, ...] = (
    "id",
    "session_id",
    "name",
    "content",
    "domain",
    "tags",
    "valence",
    "arousal",
    "importance",
    "bucket_type",
    "pinned",
    "resolved",
    "digested",
    "model_valence",
    "source_bucket_id",
    "activation_count",
    "created_at",
    "last_active_at",
)


def bucket_to_row(b: MemoryBucket) -> tuple[Any, ...]:
    """Project a :class:`MemoryBucket` into a tuple shaped for SQL binding.

    JSON columns (``domain``, ``tags``) are serialized; bool columns are
    coerced to int because SQLite has no native bool.
    """
    return (
        b.id,
        b.session_id,
        b.name,
        b.content,
        json.dumps(b.domain, ensure_ascii=False),
        json.dumps(b.tags, ensure_ascii=False),
        float(b.valence),
        float(b.arousal),
        int(b.importance),
        b.bucket_type,
        1 if b.pinned else 0,
        1 if b.resolved else 0,
        1 if b.digested else 0,
        b.model_valence,
        b.source_bucket_id,
        float(b.activation_count),
        float(b.created_at),
        float(b.last_active_at),
    )


def row_to_bucket(row: Any) -> MemoryBucket:
    """Inverse of :func:`bucket_to_row` for any aiosqlite ``Row`` mapping."""
    return MemoryBucket(
        id=row["id"],
        session_id=row["session_id"],
        name=row["name"] or "",
        content=row["content"],
        domain=_safe_json_list(row["domain"]),
        tags=_safe_json_list(row["tags"]),
        valence=float(row["valence"]),
        arousal=float(row["arousal"]),
        importance=int(row["importance"]),
        bucket_type=row["bucket_type"],
        pinned=bool(row["pinned"]),
        resolved=bool(row["resolved"]),
        digested=bool(row["digested"]),
        model_valence=(
            float(row["model_valence"]) if row["model_valence"] is not None else None
        ),
        source_bucket_id=row["source_bucket_id"],
        activation_count=float(row["activation_count"]),
        created_at=float(row["created_at"]),
        last_active_at=float(row["last_active_at"]),
    )


def _safe_json_list(raw: str | None) -> list[str]:
    """Decode a JSON list column; return ``[]`` for missing or malformed."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("malformed JSON list column, defaulting to []: %r", raw)
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class MemoryManager:
    """CRUD facade for the ``memories`` and ``embeddings`` tables.

    The manager is intentionally agnostic of LLM and embedding providers —
    those are injected by higher layers (:class:`core.memory_writer.MemoryWriter`)
    when present. The manager itself only needs the database handle.
    """

    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # CREATE
    # ------------------------------------------------------------------
    async def create(self, bucket: MemoryBucket) -> str:
        """Persist a fully-constructed bucket; returns its id.

        The caller is responsible for filling ``session_id``, ``id`` and
        timestamps — :func:`core.models.new_bucket` is the recommended
        factory. We still call ``clamp_bucket`` defensively because the
        upstream paths from LLM tools or the Dashboard may bypass the
        factory.
        """
        if not bucket.session_id:
            raise ValueError("MemoryBucket requires a non-empty session_id")
        if not bucket.id:
            raise ValueError("MemoryBucket requires a non-empty id")
        if bucket.created_at <= 0:
            now = time.time()
            bucket.created_at = now
            bucket.last_active_at = now

        clamp_bucket(bucket)

        sql = (
            "INSERT INTO memories ("
            + ", ".join(_COLUMNS)
            + ") VALUES ("
            + ", ".join("?" for _ in _COLUMNS)
            + ")"
        )
        await self.db.execute(sql, bucket_to_row(bucket))
        logger.debug("created bucket %s in session %s", bucket.id, bucket.session_id)
        return bucket.id

    async def create_simple(
        self,
        session_id: str,
        content: str,
        **kwargs: Any,
    ) -> MemoryBucket:
        """Convenience wrapper that builds a bucket with :func:`new_bucket`
        and persists it. Returns the freshly created bucket so callers do
        not need a follow-up ``get`` round-trip."""
        bucket = new_bucket(session_id=session_id, content=content, **kwargs)
        await self.create(bucket)
        return bucket

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------
    async def get(self, session_id: str, bucket_id: str) -> MemoryBucket | None:
        """Fetch a single bucket by ``(session_id, bucket_id)``.

        Querying with both keys is the cornerstone of Property 1 (session
        isolation). A bucket from another session must be invisible —
        returning it would break the privacy guarantee even if the id
        happens to be guessed.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM memories WHERE session_id = ? AND id = ?",
            (session_id, bucket_id),
        )
        return row_to_bucket(row) if row else None

    async def list_by_session(
        self,
        session_id: str,
        *,
        include_archived: bool = False,
        bucket_types: Iterable[BucketType] | None = None,
    ) -> list[MemoryBucket]:
        """List all buckets within a session, optionally filtered by type."""
        clauses: list[str] = ["session_id = ?"]
        params: list[Any] = [session_id]

        if bucket_types is not None:
            allowed = [t for t in bucket_types if t in VALID_BUCKET_TYPES]
            if not allowed:
                return []
            placeholders = ", ".join("?" for _ in allowed)
            clauses.append(f"bucket_type IN ({placeholders})")
            params.extend(allowed)
        elif not include_archived:
            clauses.append("bucket_type != 'archived'")

        sql = (
            "SELECT * FROM memories WHERE "
            + " AND ".join(clauses)
            + " ORDER BY last_active_at DESC"
        )
        rows = await self.db.fetch_all(sql, params)
        return [row_to_bucket(r) for r in rows]

    async def list_sessions(self) -> list[str]:
        """Return the distinct ``session_id`` values currently in storage.

        Used by the decay engine to iterate per session, and by the
        Dashboard's Sessions tab.
        """
        rows = await self.db.fetch_all(
            "SELECT DISTINCT session_id FROM memories ORDER BY session_id"
        )
        return [r["session_id"] for r in rows]

    async def count_in_session(self, session_id: str) -> dict[str, int]:
        """Counts per ``bucket_type`` for the given session.

        Returns a dict with keys ``dynamic / permanent / feel / archived``;
        missing types resolve to 0.
        """
        rows = await self.db.fetch_all(
            "SELECT bucket_type, COUNT(*) AS n FROM memories "
            "WHERE session_id = ? GROUP BY bucket_type",
            (session_id,),
        )
        counts: dict[str, int] = dict.fromkeys(VALID_BUCKET_TYPES, 0)
        for r in rows:
            counts[r["bucket_type"]] = int(r["n"])
        return counts

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------
    # Whitelist of fields that callers may patch via ``update``. Storage
    # columns absent from this set (id, session_id, created_at) are owned
    # by the manager itself and must not be touched externally.
    _UPDATABLE_FIELDS: frozenset[str] = frozenset(
        {
            "name",
            "content",
            "domain",
            "tags",
            "valence",
            "arousal",
            "importance",
            "bucket_type",
            "pinned",
            "resolved",
            "digested",
            "model_valence",
            "source_bucket_id",
            "activation_count",
            "last_active_at",
        }
    )

    async def update(
        self, session_id: str, bucket_id: str, **fields: Any
    ) -> MemoryBucket | None:
        """Apply a partial update; returns the updated bucket or ``None``.

        Implementation chooses load-modify-save instead of a synthesized
        UPDATE statement so that:

        - clamping rules apply uniformly (Property 11)
        - ``pinned=True`` sets ``importance=10`` and bumps the bucket type
          to ``permanent`` automatically (per the Pinned invariant)

        Unknown fields are silently ignored after a warning, so model-
        generated tool calls with typos do not crash the pipeline.
        """
        bucket = await self.get(session_id, bucket_id)
        if bucket is None:
            return None

        for key, value in fields.items():
            if key not in self._UPDATABLE_FIELDS:
                logger.warning("ignoring unknown update field: %r", key)
                continue
            setattr(bucket, key, value)

        clamp_bucket(bucket)

        # Build UPDATE statement covering only the supplied (and accepted)
        # fields plus the columns affected by clamping invariants.
        affected = {f for f in fields if f in self._UPDATABLE_FIELDS}
        # Pinned invariant may have implicitly changed importance / type.
        if "pinned" in affected:
            affected.update({"importance", "bucket_type"})

        if not affected:
            return bucket

        assignments = ", ".join(f"{col} = ?" for col in affected)
        params: list[Any] = []
        for col in affected:
            value = getattr(bucket, col)
            if col in ("domain", "tags"):
                params.append(json.dumps(value, ensure_ascii=False))
            elif col in ("pinned", "resolved", "digested"):
                params.append(1 if value else 0)
            else:
                params.append(value)
        params.extend([bucket_id, session_id])

        await self.db.execute(
            f"UPDATE memories SET {assignments} WHERE id = ? AND session_id = ?",
            params,
        )
        logger.debug(
            "updated bucket %s/%s fields=%s",
            session_id,
            bucket_id,
            sorted(affected),
        )
        return bucket

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------
    async def delete(self, session_id: str, bucket_id: str) -> bool:
        """Permanently remove a bucket and its embedding.

        SQLite ``ON DELETE CASCADE`` handles the embedding row, but we run
        both statements inside a single transaction for clarity and to
        keep behaviour stable if the foreign key is ever dropped.
        """
        async with self.db.transaction():
            cursor = await self.db._require_conn().execute(
                "DELETE FROM memories WHERE id = ? AND session_id = ?",
                (bucket_id, session_id),
            )
            deleted = cursor.rowcount or 0
            if deleted == 0:
                return False
            # Defensive cleanup; foreign key should already have done it.
            await self.db._require_conn().execute(
                "DELETE FROM embeddings WHERE bucket_id = ?",
                (bucket_id,),
            )
        logger.debug("deleted bucket %s/%s", session_id, bucket_id)
        return True

    async def archive(self, session_id: str, bucket_id: str) -> bool:
        """Move a bucket to ``bucket_type='archived'`` (decay's soft delete).

        Used by the decay engine when a score drops below threshold.
        Returns ``False`` if the bucket no longer exists.
        """
        result = await self.update(session_id, bucket_id, bucket_type="archived")
        return result is not None

    # ------------------------------------------------------------------
    # TOUCH + TIME RIPPLE
    # ------------------------------------------------------------------
    async def touch(self, session_id: str, bucket_id: str) -> None:
        """Mark a bucket as freshly recalled.

        Updates ``last_active_at`` to ``now()`` and bumps
        ``activation_count`` by ``+1.0``. Triggers :meth:`time_ripple` so
        neighbouring memories within the time window also get a small
        boost — this models the human "while you remember A, B from the
        same week subtly resurfaces too" effect (the lesson Ombre Brain
        encoded as B-03's float ``activation_count``).

        Per Property 9 this method touches only ``last_active_at`` and
        ``activation_count`` — every other field is left untouched.
        """
        now = time.time()
        cursor = await self.db.execute(
            "UPDATE memories SET last_active_at = ?, "
            "activation_count = activation_count + 1.0 "
            "WHERE id = ? AND session_id = ?",
            (now, bucket_id, session_id),
        )
        if (cursor.rowcount or 0) == 0:
            return
        # Read back the bucket's created_at so the ripple uses the right
        # reference timestamp (we ripple around when the bucket was
        # CREATED, not when it was last touched, so a bucket that's been
        # active recently doesn't pull in unrelated old neighbours).
        row = await self.db.fetch_one(
            "SELECT created_at FROM memories WHERE id = ? AND session_id = ?",
            (bucket_id, session_id),
        )
        if row is None:
            return
        try:
            await self.time_ripple(session_id, bucket_id, float(row["created_at"]))
        except Exception as e:
            # Ripple is a nice-to-have; never let it break a touch.
            logger.warning("time ripple failed for %s: %s", bucket_id, e)

    async def time_ripple(
        self,
        session_id: str,
        source_id: str,
        reference_time: float,
        *,
        window_hours: float = TIME_RIPPLE_WINDOW_HOURS,
        increment: float = TIME_RIPPLE_INCREMENT,
        max_buckets: int = TIME_RIPPLE_MAX_BUCKETS,
    ) -> int:
        """Bump ``activation_count`` for buckets created near ``reference_time``.

        Returns the number of buckets actually rippled. Hard rules:

        - never ripple ``permanent`` / ``feel`` / pinned buckets — those
          have fixed scores and should not respond to nearby activity
        - skip the source bucket itself
        - cap at ``max_buckets`` per call to bound I/O
        - this is a single UPDATE with a subquery so the manager makes one
          round-trip to SQLite regardless of window size
        """
        window_seconds = window_hours * 3600.0
        low = reference_time - window_seconds
        high = reference_time + window_seconds

        await self.db.execute(
            """
            UPDATE memories
               SET activation_count = activation_count + ?
             WHERE id IN (
                SELECT id FROM memories
                 WHERE session_id = ?
                   AND id != ?
                   AND bucket_type = 'dynamic'
                   AND pinned = 0
                   AND created_at BETWEEN ? AND ?
                 ORDER BY ABS(created_at - ?) ASC
                 LIMIT ?
             )
            """,
            (
                increment,
                session_id,
                source_id,
                low,
                high,
                reference_time,
                max_buckets,
            ),
        )
        # Reading rowcount on a CTE / subquery UPDATE is unreliable across
        # SQLite versions, so we re-query the count for accuracy in tests.
        row = await self.db.fetch_one(
            """
            SELECT COUNT(*) AS n FROM memories
             WHERE session_id = ?
               AND id != ?
               AND bucket_type = 'dynamic'
               AND pinned = 0
               AND created_at BETWEEN ? AND ?
            """,
            (session_id, source_id, low, high),
        )
        return min(int(row["n"] if row else 0), max_buckets)
