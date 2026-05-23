"""Embedding service: thin wrapper over AstrBot's EmbeddingProvider registry.

Responsibilities:
- Resolve the active embedding provider (configured id, or first available)
- Truncate text to a sane length before calling the provider (long texts
  blow through token limits and add latency without improving relevance)
- Pack/unpack vectors as little-endian float32 BLOBs suitable for SQLite
- Compute cosine similarity in pure Python (no NumPy dependency for the
  base path; we may add NumPy later if profiling demands it)

The service is deliberately stateless beyond ``self.provider``: every
method either reads from the provider, the database, or returns derived
data. This keeps testing easy — pass a fake provider that yields fixed
vectors and the rest of the pipeline behaves identically.
"""

from __future__ import annotations

import logging
import math
import struct
import time
from typing import TYPE_CHECKING, Any

from ..storage.db import Database

if TYPE_CHECKING:
    # Type-only imports; we never want a hard dependency on AstrBot at
    # import time so the unit tests can exercise this module standalone.
    from astrbot.core.provider.provider import EmbeddingProvider


logger = logging.getLogger("astrbot_plugin_ob_memory.embedding")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
MAX_INPUT_CHARS: int = 2000
"""Truncate inputs to this many characters before embedding. Matches Ombre Brain."""

DEFAULT_DIM: int = 0
"""Sentinel for "dim is unknown until first call returns"."""


# ---------------------------------------------------------------------------
# Vector packing
# ---------------------------------------------------------------------------
def pack_vector(vec: list[float]) -> bytes:
    """Serialize a vector as little-endian float32 bytes.

    float32 is enough precision for cosine similarity comparisons and uses
    half the storage of float64. Format string ``"<{n}f"`` produces a
    fixed-width buffer that we can index into without parsing.
    """
    n = len(vec)
    if n == 0:
        return b""
    return struct.pack(f"<{n}f", *vec)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    """Inverse of :func:`pack_vector`. ``dim`` must match the producer."""
    if dim <= 0 or not blob:
        return []
    expected = dim * 4
    if len(blob) != expected:
        logger.warning(
            "embedding blob size mismatch: dim=%d expected=%d got=%d",
            dim,
            expected,
            len(blob),
        )
        return []
    return list(struct.unpack(f"<{dim}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 on shape mismatch / zero vec."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class EmbeddingService:
    """Embed-and-store helper used by :class:`core.memory_writer.MemoryWriter`.

    Responsibilities:
    - resolution of the active embedding provider
    - text → vector with input truncation and graceful disable
    - vector ↔ SQLite BLOB serialisation
    - cosine search returning ``[(bucket_id, similarity), ...]``

    The class is intentionally tolerant of a missing provider:
    :meth:`embed` returns ``None``, :meth:`search_similar` returns an empty
    list, and the rest of the pipeline must already cope with that
    (embedding fallback to keyword-only search).
    """

    def __init__(
        self,
        db: Database,
        provider: EmbeddingProvider | None = None,
        *,
        context: Any = None,
        provider_id: str = "",
    ):
        self.db = db
        self.provider = provider
        # Keep context reference for lazy provider resolution — if the
        # provider wasn't available at init time (startup race), we retry
        # on first use.
        self._context = context
        self._provider_id = provider_id
        # Lazy-resolved on first successful call. ``provider.get_dim()``
        # often returns 0 until a config value is read, so we cache the
        # real dimension after the first embedding round-trip.
        self._dim: int = DEFAULT_DIM

    # ------------------------------------------------------------------
    # Resolution helpers — usable from MemoryPlugin during initialize()
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_provider(
        context: Any,
        provider_id: str = "",
    ) -> EmbeddingProvider | None:
        """Look up the embedding provider by id, falling back to the first registered.

        ``context`` is the AstrBot ``Context`` object. We accept ``Any``
        in the signature so the module can still be imported in unit
        tests where a real Context is not available.
        """
        if context is None:
            return None
        try:
            if provider_id:
                prov = context.get_provider_by_id(provider_id)
                if prov is None:
                    logger.warning(
                        "embedding provider %r not found, falling back to default",
                        provider_id,
                    )
                else:
                    return prov
            # Fall back to first registered embedding provider
            providers = context.get_all_embedding_providers()
            if providers:
                return providers[0]
        except Exception as e:
            logger.warning("failed to resolve embedding provider: %s", e)
        return None

    @property
    def enabled(self) -> bool:
        """Whether embedding calls will actually go to a provider.

        On first access when ``self.provider`` is None, attempts a lazy
        re-resolution via the stored context. This handles the AstrBot
        startup race where the embedding provider registers after the
        memory plugin's ``initialize()`` runs.
        """
        if self.provider is not None:
            return True
        # Lazy retry: provider might have become available since init
        if self._context is not None:
            self.provider = self.resolve_provider(
                self._context, provider_id=self._provider_id
            )
            if self.provider is not None:
                logger.info("embedding provider resolved on lazy retry")
                return True
        return False

    @property
    def dim(self) -> int:
        """Cached vector dimension; ``0`` if not yet known."""
        if self._dim > 0:
            return self._dim
        if self.provider is not None:
            try:
                d = int(self.provider.get_dim())
            except Exception:
                d = 0
            if d > 0:
                self._dim = d
                return d
        return DEFAULT_DIM

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    async def embed(self, text: str) -> list[float] | None:
        """Embed ``text`` via the configured provider.

        Returns the vector on success, ``None`` if disabled or failed.
        Truncates input to ``MAX_INPUT_CHARS`` to avoid breaching token
        limits on long content.
        """
        if not self.enabled or not text:
            return None
        truncated = text[:MAX_INPUT_CHARS]
        try:
            vec = await self.provider.get_embedding(truncated)  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("embedding API call failed: %s", e)
            return None
        if not vec:
            return None
        if self._dim == 0:
            self._dim = len(vec)
        return list(vec)

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        """Embed ``content`` and persist the vector into ``embeddings``.

        Returns ``True`` if a vector was successfully stored. Idempotent:
        re-storing for the same ``bucket_id`` overwrites the prior row.
        """
        vec = await self.embed(content)
        if vec is None:
            return False
        await self._store(bucket_id, vec)
        return True

    async def _store(self, bucket_id: str, vec: list[float]) -> None:
        """Write a packed vector to the ``embeddings`` table."""
        await self.db.execute(
            "INSERT OR REPLACE INTO embeddings (bucket_id, vector, dim, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (bucket_id, pack_vector(vec), len(vec), time.time()),
        )

    async def get(self, bucket_id: str) -> list[float] | None:
        """Read back a stored vector, or ``None`` if absent or malformed."""
        row = await self.db.fetch_one(
            "SELECT vector, dim FROM embeddings WHERE bucket_id = ?",
            (bucket_id,),
        )
        if row is None:
            return None
        vec = unpack_vector(row["vector"], int(row["dim"]))
        return vec or None

    async def delete(self, bucket_id: str) -> None:
        """Remove the stored vector, if any.

        ``MemoryManager.delete`` already triggers ON DELETE CASCADE — this
        method exists for cases where a refresh wants to purge the row
        without removing the bucket itself.
        """
        await self.db.execute(
            "DELETE FROM embeddings WHERE bucket_id = ?", (bucket_id,)
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    async def search_similar(
        self,
        session_id: str,
        query: str,
        *,
        top_k: int = 50,
        min_similarity: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Find the top-K most similar buckets within ``session_id``.

        The search is naive (load all vectors for the session, score in
        Python). For a typical session size (≤ a few thousand buckets)
        this is well under 10ms and avoids any C extension or external
        service. We can add an HNSW / FAISS index later if profiling
        ever shows it as a hotspot.
        """
        query_vec = await self.embed(query)
        if query_vec is None:
            return []

        rows = await self.db.fetch_all(
            """
            SELECT e.bucket_id, e.vector, e.dim
              FROM embeddings AS e
              JOIN memories AS m ON m.id = e.bucket_id
             WHERE m.session_id = ?
            """,
            (session_id,),
        )
        if not rows:
            return []

        scored: list[tuple[str, float]] = []
        for row in rows:
            stored = unpack_vector(row["vector"], int(row["dim"]))
            if not stored:
                continue
            sim = cosine_similarity(query_vec, stored)
            if sim >= min_similarity:
                scored.append((row["bucket_id"], sim))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]
