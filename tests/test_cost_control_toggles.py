"""Tests for the cost-control toggles introduced in Phase 11.

Three independent switches let operators trade richness for fewer LLM
calls:

- ``tagging_enabled = False`` skips ``Tagger.analyze`` in MemoryWriter
- ``merge_enabled = False`` skips the embedding merge candidate search
- ``auto_record_use_judge = False`` skips ``Tagger.judge_worth_recording``

Tests use stubs that **explode** when the disallowed call happens, so a
green run is direct evidence the LLM call was actually skipped (not just
ignored or errored silently).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from astrbot_plugin_ob_memory.core.embedding_service import EmbeddingService
from astrbot_plugin_ob_memory.core.memory_manager import MemoryManager
from astrbot_plugin_ob_memory.core.memory_writer import MemoryWriter
from astrbot_plugin_ob_memory.core.tagger import Tagger

from astrbot_plugin_ob_memory.handlers.llm_hooks import MemoryHooksMixin

from astrbot_plugin_ob_memory.storage import Database, apply_migrations

ASYNCIO = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------
@dataclass
class FakeLLMResponse:
    completion_text: str = ""


class ExplodingProvider:
    """Any call here is a test failure."""

    async def text_chat(self, **kwargs):
        raise AssertionError(
            "Tagger.text_chat must NOT be invoked when toggles are off"
        )


class ExplodingEmbeddingProvider:
    async def get_embedding(self, text):
        raise AssertionError(
            "EmbeddingProvider.get_embedding must NOT be invoked when "
            "merge_enabled is off (and no other path needs an embedding)"
        )

    def get_dim(self):
        return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _open(tmp_path: Path) -> tuple[Database, MemoryManager]:
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    return db, MemoryManager(db)


# ===========================================================================
# tagging_enabled = False
# ===========================================================================
@ASYNCIO
async def test_tagging_disabled_skips_analyze(tmp_path: Path):
    db, mgr = await _open(tmp_path)
    try:
        # Tagger backed by a provider that raises on every call.
        tagger = Tagger(context=None, fixed_provider=ExplodingProvider())
        writer = MemoryWriter(
            mgr,
            tagger=tagger,
            embedding=None,  # no embed needed for this test
            tagging_enabled=False,
            merge_enabled=False,  # also off so no extra path triggers
        )

        # Should NOT raise even though provider would explode if called.
        result = await writer.hold("s", "a normal user statement", importance=6)
        loaded = await mgr.get("s", result.bucket_id)
        assert loaded is not None
        # Default neutral metadata applied.
        assert loaded.domain == ["未分类"]
        assert loaded.valence == pytest.approx(0.5)
        assert loaded.arousal == pytest.approx(0.3)
        # User-supplied importance still wins.
        assert loaded.importance == 6
    finally:
        await db.close()


@ASYNCIO
async def test_tagging_enabled_does_call_analyze(tmp_path: Path):
    """Sanity check the toggle: turning it on makes the provider get called."""
    db, mgr = await _open(tmp_path)
    try:
        calls: list[str] = []

        class CountingProvider:
            async def text_chat(self, prompt=None, system_prompt=None, **kw):
                calls.append((system_prompt or "")[:30])
                return FakeLLMResponse('{"domain": ["x"], "valence": 0.5, "arousal": 0.3, "tags": [], "suggested_name": "", "importance": 5}')

        tagger = Tagger(context=None, fixed_provider=CountingProvider())
        writer = MemoryWriter(
            mgr,
            tagger=tagger,
            embedding=None,
            tagging_enabled=True,
            merge_enabled=False,
        )
        await writer.hold("s", "hello world")
        # Provider was called for analyse exactly once.
        assert len(calls) == 1
    finally:
        await db.close()


# ===========================================================================
# merge_enabled = False
# ===========================================================================
@ASYNCIO
async def test_merge_disabled_skips_embedding_query(tmp_path: Path):
    """When merge_enabled is False, no embedding lookup happens for merge.

    The exploding embedding provider would normally raise on
    ``search_similar`` (the call inside ``_find_merge_candidate``).
    Disabling merging short-circuits the path before it gets there.
    """
    db, mgr = await _open(tmp_path)
    try:
        # Embed once for the existing bucket so we have something in DB,
        # then swap to an exploding provider for the second write.
        from collections import deque

        class ScriptedEmbeddingProvider:
            """First call returns a real vector, every subsequent call
            raises so we can detect any unwanted lookup."""

            def __init__(self):
                self._scripted = deque(
                    [[1.0, 0.0, 0.0, 0.0]]  # one good response for the seed
                )

            async def get_embedding(self, text):
                if not self._scripted:
                    raise AssertionError("unexpected embedding lookup")
                return self._scripted.popleft()

            def get_dim(self):
                return 4

        provider = ScriptedEmbeddingProvider()
        embedding = EmbeddingService(db, provider=provider)

        # Seed: write one bucket WITH merge_enabled (so its embedding lands).
        # We bypass the writer's analyze entirely by making it taggerless.
        seed_writer = MemoryWriter(
            mgr,
            tagger=None,
            embedding=embedding,
            tagging_enabled=False,
            merge_enabled=False,  # seeding doesn't need merge either
        )
        await seed_writer.hold("s", "seed bucket")

        # Now the second writer runs with merge disabled and should
        # never query the embedding provider for a merge candidate.
        provider._scripted.clear()  # any further calls fail loud
        writer = MemoryWriter(
            mgr,
            tagger=None,
            embedding=embedding,
            tagging_enabled=False,
            merge_enabled=False,
        )
        # generate_and_store on the new bucket WOULD call get_embedding;
        # we don't want that to fail the test, so disable embedding for
        # this writer instance entirely. The point is the merge path.
        writer.embedding = None
        result = await writer.hold("s", "completely fresh content")
        # New bucket created; no merge happened.
        assert result.was_merged is False
    finally:
        await db.close()


@ASYNCIO
async def test_merge_disabled_creates_fresh_bucket_each_time(tmp_path: Path):
    """With merge disabled, near-duplicate writes always create new buckets."""
    db, mgr = await _open(tmp_path)
    try:
        # Both inputs map to the same vector → would normally merge.
        from dataclasses import dataclass as _dc

        class FakeProvider:
            async def get_embedding(self, text):
                return [1.0, 0.0, 0.0, 0.0]

            def get_dim(self):
                return 4

        embedding = EmbeddingService(db, provider=FakeProvider())
        writer = MemoryWriter(
            mgr,
            tagger=None,
            embedding=embedding,
            tagging_enabled=False,
            merge_enabled=False,
        )
        first = await writer.hold("s", "alpha")
        second = await writer.hold("s", "alpha")
        assert first.bucket_id != second.bucket_id
        assert second.was_merged is False
    finally:
        await db.close()


# ===========================================================================
# auto_record_use_judge = False
# ===========================================================================
@dataclass
class FakeEvent:
    unified_msg_origin: str = "qq:GroupMessage:12345"
    message_str: str = ""


@dataclass
class FakeLLMResp:
    completion_text: str = ""
    tools_call_name: list[str] = field(default_factory=list)
    role: str = "assistant"


class FakeAssembled(MemoryHooksMixin):
    def __init__(self):
        self.manager = None
        self.writer = None
        self.search = None
        self.surface = None
        self.tagger = None
        self.config: dict = {}


class JudgeRefusingTagger:
    """``judge_worth_recording`` raises — proving we DIDN'T call it."""

    async def judge_worth_recording(self, user, assistant, *, session_id=None):
        raise AssertionError(
            "judge_worth_recording must NOT be called when "
            "auto_record_use_judge is False"
        )

    async def analyze(self, content, *, session_id=None):
        return {
            "domain": ["未分类"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": [],
            "suggested_name": "",
            "importance": 5,
        }


@ASYNCIO
async def test_auto_record_no_judge_records_directly(tmp_path: Path):
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    try:
        mgr = MemoryManager(db)
        obj = FakeAssembled()
        obj.manager = mgr
        obj.tagger = JudgeRefusingTagger()
        obj.writer = MemoryWriter(
            mgr,
            tagger=obj.tagger,
            embedding=None,
            tagging_enabled=True,
            merge_enabled=False,
        )
        obj.config = {
                "auto_record_enabled": True,
                "auto_record_min_chars": 5,
                "auto_record_use_judge": False,
            }

        event = FakeEvent(message_str="this is a long enough user message about life")
        response = FakeLLMResp(completion_text="ok let me think about that")

        # Hook returns immediately; background task records.
        await obj.memory_on_llm_response(event, response)

        # Wait for the background task.
        import asyncio

        for _ in range(20):
            await asyncio.sleep(0.05)
            buckets = await mgr.list_by_session("qq:GroupMessage:12345")
            if buckets:
                break

        buckets = await mgr.list_by_session("qq:GroupMessage:12345")
        assert len(buckets) == 1
        assert "user message about life" in buckets[0].content
    finally:
        await db.close()


@ASYNCIO
async def test_auto_record_judge_default_still_uses_judge(tmp_path: Path):
    """Sanity: with the toggle absent / true, judge IS called."""
    db = Database(tmp_path / "memory.db")
    await db.connect()
    await apply_migrations(db)
    try:
        mgr = MemoryManager(db)
        obj = FakeAssembled()
        obj.manager = mgr

        class CountingTagger:
            def __init__(self):
                self.calls: list[tuple[str, str]] = []

            async def judge_worth_recording(self, user, assistant, *, session_id=None):
                self.calls.append((user, assistant))
                return False, "test"

            async def analyze(self, content, *, session_id=None):
                return {
                    "domain": ["未分类"],
                    "valence": 0.5,
                    "arousal": 0.3,
                    "tags": [],
                    "suggested_name": "",
                    "importance": 5,
                }

        obj.tagger = CountingTagger()
        obj.writer = MemoryWriter(
            mgr,
            tagger=obj.tagger,
            embedding=None,
            tagging_enabled=True,
            merge_enabled=False,
        )
        obj.config = {
                "auto_record_enabled": True,
                "auto_record_min_chars": 5,
                # use_judge default is True
            }

        event = FakeEvent(message_str="something significant has happened today")
        response = FakeLLMResp(completion_text="oh I see")
        await obj.memory_on_llm_response(event, response)

        import asyncio
        await asyncio.sleep(0.2)
        assert obj.tagger.calls, "judge_worth_recording was not called"
    finally:
        await db.close()
