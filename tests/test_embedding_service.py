"""Unit tests for ``core.embedding_service``.

Covers vector packing/unpacking, cosine similarity, and the search +
storage flow against a stub embedding provider.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from astrbot_plugin_ob_memory.core.embedding_service import (
    EmbeddingService,
    cosine_similarity,
    pack_vector,
    unpack_vector,
)
from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.storage import Database, apply_migrations

# Module-level asyncio mark would also catch synchronous helper tests below
# and trigger a noisy PytestWarning. We mark the async tests individually.
ASYNCIO = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeEmbeddingProvider:
    """Minimal stand-in for ``EmbeddingProvider``.

    Returns deterministic vectors so similarity assertions are stable. The
    fake stores a ``calls`` log so tests can verify exact invocations.
    """

    def __init__(self, mapping: dict[str, list[float]] | None = None, dim: int = 4):
        self._mapping = mapping or {}
        self._dim = dim
        self.calls: list[str] = []

    async def get_embedding(self, text: str) -> list[float]:
        self.calls.append(text)
        if text in self._mapping:
            return list(self._mapping[text])
        # Deterministic default: tiny vector seeded from char codes.
        seed = sum(ord(c) for c in text) % 97
        return [
            (seed + i) / 100.0
            for i in range(self._dim)
        ]

    def get_dim(self) -> int:
        return self._dim


class ExplodingEmbeddingProvider:
    """Always raises — to exercise the graceful-disable path."""

    async def get_embedding(self, text: str) -> list[float]:
        raise RuntimeError("provider down")

    def get_dim(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Pure utilities
# ---------------------------------------------------------------------------
def test_pack_unpack_round_trip():
    vec = [0.1, -0.5, 1.5, 3.14, -2.71]
    blob = pack_vector(vec)
    restored = unpack_vector(blob, len(vec))
    for a, b in zip(vec, restored, strict=True):
        # float32 has limited precision; just need rough equality.
        assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)


def test_pack_empty_returns_empty_blob():
    assert pack_vector([]) == b""


def test_unpack_handles_size_mismatch():
    blob = pack_vector([1.0, 2.0, 3.0])
    # Claim a different dim — should return [] and not raise.
    assert unpack_vector(blob, 4) == []


def test_cosine_identical():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_opposite():
    v = [1.0, 0.0]
    w = [-1.0, 0.0]
    assert cosine_similarity(v, w) == pytest.approx(-1.0)


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_shape_mismatch():
    assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0


# ---------------------------------------------------------------------------
# Service: enabled / disabled paths
# ---------------------------------------------------------------------------
@ASYNCIO
async def test_disabled_when_no_provider(tmp_path: Path):
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    try:
        svc = EmbeddingService(db, provider=None)
        assert svc.enabled is False
        assert await svc.embed("anything") is None
        assert await svc.generate_and_store("id-1", "x") is False
        # Search must also degrade gracefully.
        assert await svc.search_similar("session-A", "query") == []
    finally:
        await db.close()


@ASYNCIO
async def test_embed_truncates_long_input(tmp_path: Path):
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    try:
        provider = FakeEmbeddingProvider()
        svc = EmbeddingService(db, provider=provider)
        long_text = "a" * 5000
        vec = await svc.embed(long_text)
        assert vec is not None
        assert len(vec) == provider.get_dim()
        # Provider should have seen a truncated input.
        assert len(provider.calls) == 1
        assert len(provider.calls[0]) <= 2000
    finally:
        await db.close()


@ASYNCIO
async def test_provider_error_returns_none(tmp_path: Path):
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    try:
        svc = EmbeddingService(db, provider=ExplodingEmbeddingProvider())
        assert svc.enabled is True
        assert await svc.embed("anything") is None
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Service: store + retrieve
# ---------------------------------------------------------------------------
@ASYNCIO
async def test_generate_and_store_round_trip(tmp_path: Path):
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    try:
        provider = FakeEmbeddingProvider(mapping={"hello": [0.1, 0.2, 0.3, 0.4]})
        svc = EmbeddingService(db, provider=provider)
        mgr = MemoryManager(db)
        bucket = await mgr.create_simple("session-A", "hello")

        ok = await svc.generate_and_store(bucket.id, "hello")
        assert ok is True

        restored = await svc.get(bucket.id)
        assert restored is not None
        assert len(restored) == 4
    finally:
        await db.close()


@ASYNCIO
async def test_search_similar_ranks_by_cosine(tmp_path: Path):
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    try:
        # Three buckets with hand-picked vectors. Query is identical to A.
        mapping = {
            "bucket-A-text": [1.0, 0.0, 0.0, 0.0],   # similarity 1.0 with query
            "bucket-B-text": [0.6, 0.8, 0.0, 0.0],   # ~0.6
            "bucket-C-text": [0.0, 1.0, 0.0, 0.0],   # 0.0 (orthogonal → filtered out)
            "query": [1.0, 0.0, 0.0, 0.0],
        }
        provider = FakeEmbeddingProvider(mapping=mapping)
        svc = EmbeddingService(db, provider=provider)
        mgr = MemoryManager(db)

        a = await mgr.create_simple("session-A", "bucket-A-text")
        b = await mgr.create_simple("session-A", "bucket-B-text")
        c = await mgr.create_simple("session-A", "bucket-C-text")
        await svc.generate_and_store(a.id, "bucket-A-text")
        await svc.generate_and_store(b.id, "bucket-B-text")
        await svc.generate_and_store(c.id, "bucket-C-text")

        results = await svc.search_similar("session-A", "query", min_similarity=0.5)
        ids = [r[0] for r in results]
        assert ids[0] == a.id
        assert b.id in ids
        assert c.id not in ids  # below the 0.5 threshold
    finally:
        await db.close()


@ASYNCIO
async def test_search_isolates_sessions(tmp_path: Path):
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    try:
        provider = FakeEmbeddingProvider(
            mapping={
                "shared": [1.0, 0.0],
                "query": [1.0, 0.0],
            },
            dim=2,
        )
        svc = EmbeddingService(db, provider=provider)
        mgr = MemoryManager(db)

        a = await mgr.create_simple("session-A", "shared")
        b = await mgr.create_simple("session-B", "shared")
        await svc.generate_and_store(a.id, "shared")
        await svc.generate_and_store(b.id, "shared")

        a_results = await svc.search_similar("session-A", "query")
        b_results = await svc.search_similar("session-B", "query")
        assert [r[0] for r in a_results] == [a.id]
        assert [r[0] for r in b_results] == [b.id]
    finally:
        await db.close()


@ASYNCIO
async def test_delete_removes_only_target(tmp_path: Path):
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    try:
        provider = FakeEmbeddingProvider()
        svc = EmbeddingService(db, provider=provider)
        mgr = MemoryManager(db)
        a = await mgr.create_simple("s", "alpha")
        b = await mgr.create_simple("s", "bravo")
        await svc.generate_and_store(a.id, "alpha")
        await svc.generate_and_store(b.id, "bravo")

        await svc.delete(a.id)
        assert await svc.get(a.id) is None
        assert await svc.get(b.id) is not None
    finally:
        await db.close()
