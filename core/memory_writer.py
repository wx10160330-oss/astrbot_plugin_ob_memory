"""High-level memory write flows: ``hold`` and ``hold_feel``.

The MemoryManager gives us low-level CRUD over the SQLite tables. Real
writes happen through THIS module, which composes:

- :class:`core.tagger.Tagger` — auto-analyse content into Russell
  coordinates, domain, tags, suggested_name
- :class:`core.embedding_service.EmbeddingService` — generate the
  embedding, find merge candidates
- :class:`core.memory_manager.MemoryManager` — final CRUD

Two flows are exposed:

- :meth:`hold` — store a normal event memory. Auto-tags via LLM, decides
  merge-vs-create via embedding similarity, refreshes the merged bucket
  on hit. Returns ``(bucket_id, was_merged)``.
- :meth:`hold_feel` — store a model-perspective reflection. Skips
  auto-tagging and merging entirely; if a ``source_bucket_id`` is given
  the source bucket is marked ``digested=True``.

Both flows are tolerant of missing services. With no Tagger / no LLM
provider, ``hold`` still creates a bucket with default metadata. With no
EmbeddingService, merge detection silently degrades and every write goes
to a new bucket. This means a bare-bones AstrBot install (no embedding
provider configured, no LLM key) can still record memories — they just
have less rich metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .embedding_service import EmbeddingService
from .memory_manager import MemoryManager
from .models import MemoryBucket, clamp_bucket, new_bucket
from .tagger import Tagger

logger = logging.getLogger("astrbot_plugin_ob_memory.writer")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DEFAULT_MERGE_THRESHOLD: float = 0.85
"""Cosine-similarity floor for merging into an existing bucket.

0.85 is conservative — only near-duplicates get merged. Higher values
would prevent legitimate consolidation, lower values would cause
unrelated content to bleed into the same bucket.
"""


@dataclass(frozen=True)
class HoldResult:
    """Outcome of a ``hold`` call.

    Returned as a structured value (rather than a 2-tuple) so future
    additions — e.g. analyse-time confidence, merged-into bucket id —
    don't change the call site signatures.
    """

    bucket_id: str
    was_merged: bool
    target_bucket: MemoryBucket
    """The bucket that ended up holding the content (whether new or merged-into)."""


@dataclass(frozen=True)
class FeelResult:
    """Outcome of a ``hold_feel`` call."""

    bucket_id: str
    source_marked_digested: bool


@dataclass(frozen=True)
class DigestResult:
    """Outcome of a ``hold_diary`` call.

    Aggregates the per-entry results when a long passage is split into
    multiple memories. ``entries`` is the list of created/merged buckets
    in the order they were processed; ``failed`` counts entries that
    raised mid-flow (single-entry isolation).
    """

    entries: list[HoldResult]
    failed: int

    @property
    def created(self) -> int:
        return sum(1 for h in self.entries if not h.was_merged)

    @property
    def merged(self) -> int:
        return sum(1 for h in self.entries if h.was_merged)


class MemoryWriter:
    """Orchestrator for high-level memory writes.

    A writer is intentionally cheap to construct (just refs) so the
    plugin can rebuild it whenever the active LLM / embedding provider
    changes — for example after AstrBot's per-session provider swap.
    """

    def __init__(
        self,
        manager: MemoryManager,
        *,
        tagger: Tagger | None = None,
        embedding: EmbeddingService | None = None,
        merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
        tagging_enabled: bool = True,
        merge_enabled: bool = True,
        digest_prompt: str = "",
    ):
        self.manager = manager
        self.tagger = tagger
        self.embedding = embedding
        self.merge_threshold = merge_threshold
        # Cost-control toggles. Reset by ``MemoryPlugin._refresh_writer_toggles``
        # whenever config is reloaded; tests can poke them directly.
        self.tagging_enabled = tagging_enabled
        self.merge_enabled = merge_enabled
        # User-customisable digest prompt (empty = use built-in default)
        self.digest_prompt = digest_prompt

    # ------------------------------------------------------------------
    # hold (event memory)
    # ------------------------------------------------------------------
    async def hold(
        self,
        session_id: str,
        content: str,
        *,
        importance: int | None = None,
        tags: list[str] | None = None,
        pinned: bool = False,
        valence: float | None = None,
        arousal: float | None = None,
    ) -> HoldResult:
        """Store a piece of event content as a memory bucket.

        The ``user-supplied valence / arousal / importance`` arguments take
        precedence over Tagger output. This matters for the LLM tool path:
        when the model passes explicit values, those are its considered
        judgement and should not be overwritten by a second analyse pass.
        """
        if not content or not content.strip():
            raise ValueError("hold(content) requires non-empty content")

        # ---------- 1. Auto-analyse ----------
        # Skip the LLM analyse call when tagging is disabled or no Tagger
        # is bound. Either way we degrade to neutral defaults so the rest
        # of the flow sees a uniform shape.
        analysis: dict
        if self.tagger is None or not self.tagging_enabled:
            from .tagger import DEFAULT_ANALYZE

            analysis = dict(DEFAULT_ANALYZE)
        else:
            analysis = await self.tagger.analyze(content, session_id=session_id)

        return await self._persist_with_analysis(
            session_id,
            content,
            analysis=analysis,
            importance=importance,
            tags=tags,
            pinned=pinned,
            valence=valence,
            arousal=arousal,
        )

    async def _persist_with_analysis(
        self,
        session_id: str,
        content: str,
        *,
        analysis: dict,
        importance: int | None = None,
        tags: list[str] | None = None,
        pinned: bool = False,
        valence: float | None = None,
        arousal: float | None = None,
    ) -> HoldResult:
        """Store ``content`` using a pre-computed ``analysis`` dict.

        Internal helper used by :meth:`hold` (which always analyses) and
        :meth:`hold_diary` (which gets per-entry analyses from
        :meth:`Tagger.digest`, saving an LLM call per entry).
        """
        # User overrides win when within range; analysis fills the rest.
        final_valence = valence if valence is not None else analysis["valence"]
        final_arousal = arousal if arousal is not None else analysis["arousal"]
        final_importance = (
            importance if importance is not None else int(analysis["importance"])
        )

        # Combine analyser tags with user-supplied tags (deduped, order
        # preserved — user tags first since they're the explicit signal).
        user_tags = list(tags or [])
        merged_tags = list(dict.fromkeys(user_tags + list(analysis["tags"])))

        # ---------- 2. Pinned bucket: skip merge, force permanent ----------
        if pinned:
            new = new_bucket(
                session_id=session_id,
                content=content,
                name=analysis["suggested_name"] or "",
                domain=list(analysis["domain"]),
                tags=merged_tags,
                valence=final_valence,
                arousal=final_arousal,
                importance=final_importance,
                pinned=True,
            )
            await self.manager.create(new)
            await self._safe_embed(new.id, content)
            logger.debug(
                "pinned bucket created session=%s id=%s name=%r",
                session_id,
                new.id,
                new.name,
            )
            return HoldResult(bucket_id=new.id, was_merged=False, target_bucket=new)

        # ---------- 3. Merge detection (vector channel) ----------
        candidate = await self._find_merge_candidate(session_id, content)
        if candidate is not None:
            updated = await self._merge_into(
                session_id=session_id,
                target=candidate,
                new_content=content,
                new_domain=list(analysis["domain"]),
                new_tags=merged_tags,
                new_valence=final_valence,
                new_arousal=final_arousal,
                new_importance=final_importance,
            )
            return HoldResult(
                bucket_id=updated.id, was_merged=True, target_bucket=updated
            )

        # ---------- 4. Create a fresh bucket ----------
        new = new_bucket(
            session_id=session_id,
            content=content,
            name=analysis["suggested_name"] or "",
            domain=list(analysis["domain"]),
            tags=merged_tags,
            valence=final_valence,
            arousal=final_arousal,
            importance=final_importance,
        )
        await self.manager.create(new)
        await self._safe_embed(new.id, content)
        logger.debug(
            "new bucket created session=%s id=%s name=%r",
            session_id,
            new.id,
            new.name,
        )
        return HoldResult(bucket_id=new.id, was_merged=False, target_bucket=new)

    # ------------------------------------------------------------------
    # hold_feel (model reflection)
    # ------------------------------------------------------------------
    async def hold_feel(
        self,
        session_id: str,
        content: str,
        *,
        source_bucket_id: str | None = None,
        valence: float | None = None,
    ) -> FeelResult:
        """Store a feel — the model's first-person reflection on something.

        Feel buckets:
        - skip auto-tagging entirely (the content IS the model's analysis;
          re-analysing it muddies the signal)
        - skip merging (each feel is its own data point, even if similar
          to another)
        - go to ``bucket_type='feel'`` so they never participate in
          surfacing or normal decay (DecayEngine returns a fixed score of
          50.0 for them)

        If ``source_bucket_id`` is provided, that bucket is marked as
        ``digested=True`` and ``model_valence`` is recorded — this is how
        the source memory transitions from "an event I remember" to
        "an event I've digested into a feeling".
        """
        if not content or not content.strip():
            raise ValueError("hold_feel(content) requires non-empty content")

        feel_valence = valence if valence is not None else 0.5
        feel_arousal = 0.3  # feel buckets never participate in arousal-based ranking

        bucket = new_bucket(
            session_id=session_id,
            content=content,
            valence=feel_valence,
            arousal=feel_arousal,
            importance=5,
            bucket_type="feel",
        )
        bucket.model_valence = feel_valence
        bucket.source_bucket_id = (
            source_bucket_id.strip()
            if source_bucket_id and source_bucket_id.strip()
            else None
        )
        clamp_bucket(bucket)
        await self.manager.create(bucket)
        await self._safe_embed(bucket.id, content)

        marked = False
        if bucket.source_bucket_id:
            try:
                updated = await self.manager.update(
                    session_id,
                    bucket.source_bucket_id,
                    digested=True,
                    model_valence=feel_valence,
                )
                marked = updated is not None
                if not marked:
                    logger.warning(
                        "feel: source bucket %s not found in session %s",
                        bucket.source_bucket_id,
                        session_id,
                    )
            except Exception as e:
                # Failure to mark the source must not invalidate the feel.
                logger.warning(
                    "feel: failed to mark source %s as digested: %s",
                    bucket.source_bucket_id,
                    e,
                )

        return FeelResult(bucket_id=bucket.id, source_marked_digested=marked)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _find_merge_candidate(
        self, session_id: str, content: str
    ) -> MemoryBucket | None:
        """Return the highest-similarity non-pinned bucket above threshold.

        Short-circuits to ``None`` when merging is disabled or no
        embedding service is bound — keeping the merge path strictly
        opt-in for cost-conscious deployments.
        """
        if not self.merge_enabled:
            return None
        if self.embedding is None or not self.embedding.enabled:
            return None
        try:
            results = await self.embedding.search_similar(
                session_id,
                content,
                top_k=5,
                min_similarity=self.merge_threshold,
            )
        except Exception as e:
            logger.warning("merge candidate search failed: %s", e)
            return None
        for bucket_id, _sim in results:
            try:
                bucket = await self.manager.get(session_id, bucket_id)
            except Exception:
                continue
            if bucket is None:
                continue
            if bucket.pinned or bucket.bucket_type in ("permanent", "feel", "archived"):
                continue
            return bucket
        return None

    async def _merge_into(
        self,
        *,
        session_id: str,
        target: MemoryBucket,
        new_content: str,
        new_domain: list[str],
        new_tags: list[str],
        new_valence: float,
        new_arousal: float,
        new_importance: int,
    ) -> MemoryBucket:
        """Fold ``new_content`` into ``target`` and refresh the embedding."""
        if self.tagger is None:
            merged_content = f"{target.content}\n\n{new_content}".strip()
        else:
            merged_content = await self.tagger.merge_content(
                target.content, new_content, session_id=session_id
            )

        merged_tags = list(dict.fromkeys(list(target.tags) + new_tags))
        merged_domain = list(dict.fromkeys(list(target.domain) + new_domain))
        avg_valence = (target.valence + new_valence) / 2.0
        avg_arousal = (target.arousal + new_arousal) / 2.0
        max_importance = max(int(target.importance), int(new_importance))

        updated = await self.manager.update(
            session_id,
            target.id,
            content=merged_content,
            tags=merged_tags,
            domain=merged_domain,
            valence=avg_valence,
            arousal=avg_arousal,
            importance=max_importance,
        )
        if updated is None:
            # Race condition: target deleted between candidate selection and
            # the update. Fall back to a fresh create using the new values.
            logger.warning(
                "merge target %s vanished mid-flow; creating fresh bucket",
                target.id,
            )
            fresh = new_bucket(
                session_id=session_id,
                content=new_content,
                domain=new_domain,
                tags=merged_tags,
                valence=new_valence,
                arousal=new_arousal,
                importance=new_importance,
            )
            await self.manager.create(fresh)
            await self._safe_embed(fresh.id, new_content)
            return fresh

        # Refresh embedding so future searches reflect the merged content.
        await self._safe_embed(updated.id, merged_content)
        return updated

    async def _safe_embed(self, bucket_id: str, content: str) -> None:
        """Embed and store; swallow any provider error."""
        if self.embedding is None or not self.embedding.enabled:
            return
        try:
            await self.embedding.generate_and_store(bucket_id, content)
        except Exception as e:
            logger.warning("embedding generation failed for %s: %s", bucket_id, e)

    # ------------------------------------------------------------------
    # hold_diary (long-form passage → multiple memories)
    # ------------------------------------------------------------------
    SHORT_DIARY_THRESHOLD: int = 30
    """Inputs shorter than this skip the LLM digest call entirely and go
    straight through ``hold`` (saves a wasted LLM round-trip on tiny
    inputs)."""

    async def hold_diary(
        self,
        session_id: str,
        content: str,
    ) -> DigestResult:
        """Split a long passage into entries and hold each as its own memory.

        Behaviour:
        - Inputs shorter than :data:`SHORT_DIARY_THRESHOLD` chars take a
          fast path that calls :meth:`hold` directly with the whole text.
        - Otherwise, calls :meth:`Tagger.digest` to split into 2-6 entries.
        - Each entry is processed independently — a failure on one entry
          (LLM error, embedding error, race-deleted merge target) does
          NOT abort the rest.
        - Returns a :class:`DigestResult` summarising created vs merged.

        Used by the ``record_diary`` LLM tool. With no Tagger or no LLM
        provider this degrades to the single-bucket fast path.
        """
        if not content or not content.strip():
            raise ValueError("hold_diary(content) requires non-empty content")

        text = content.strip()

        # Fast path: short inputs aren't worth a digest LLM call.
        if len(text) < self.SHORT_DIARY_THRESHOLD or self.tagger is None:
            try:
                hold = await self.hold(session_id, text)
                return DigestResult(entries=[hold], failed=0)
            except Exception as e:
                logger.warning("hold_diary fast path failed: %s", e)
                return DigestResult(entries=[], failed=1)

        # Normal path: split into entries via LLM.
        try:
            entries = await self.tagger.digest(
                text,
                session_id=session_id,
                digest_prompt_override=self.digest_prompt or None,
            )
        except Exception as e:
            logger.warning(
                "hold_diary digest failed, falling back to single hold: %s", e
            )
            entries = []

        if not entries:
            # Either splitting failed or LLM returned empty — store as one.
            try:
                hold = await self.hold(session_id, text)
                return DigestResult(entries=[hold], failed=0)
            except Exception as e:
                logger.warning("hold_diary single-bucket fallback failed: %s", e)
                return DigestResult(entries=[], failed=1)

        # Hold each entry independently.
        results: list[HoldResult] = []
        failed = 0
        for entry in entries:
            try:
                # The digest analyser already supplied valence/arousal/etc.
                # We forward the entry as a pre-computed analysis dict to
                # ``_persist_with_analysis`` so we don't re-call analyze
                # (saves one LLM round-trip per entry).
                analysis = {
                    "domain": entry["domain"],
                    "valence": entry["valence"],
                    "arousal": entry["arousal"],
                    "tags": entry["tags"],
                    "suggested_name": entry.get("suggested_name", ""),
                    "importance": entry["importance"],
                }
                hold = await self._persist_with_analysis(
                    session_id,
                    entry["content"],
                    analysis=analysis,
                )
                results.append(hold)
            except Exception as e:
                logger.warning(
                    "hold_diary entry failed (name=%r): %s",
                    entry.get("suggested_name", "?"),
                    e,
                )
                failed += 1

        return DigestResult(entries=results, failed=failed)
