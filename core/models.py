"""Memory bucket data model and helper utilities.

A ``MemoryBucket`` is the unit of storage for a single piece of long-term
memory. The schema mirrors the ``memories`` SQLite table 1-to-1 so that
serialization is mechanical (see ``memory_manager.bucket_to_row``).

Design notes:
- ``valence`` and ``arousal`` follow Russell's circumplex model, both in
  the inclusive range [0.0, 1.0]. Use ``clamp_bucket`` before persistence.
- ``importance`` is in [1, 10].
- ``bucket_type`` is constrained to the ``BucketType`` Literal.
- ``activation_count`` is a float (not int) to allow the time-ripple
  mechanism to add fractional increments (e.g. 0.3) for neighbouring
  buckets recalled together.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

BucketType = Literal["dynamic", "permanent", "feel", "archived"]
"""Lifecycle classification of a bucket.

- ``dynamic``: normal event memory, subject to decay
- ``permanent``: pinned/protected, never decays
- ``feel``: model's first-person reflection, never decays, never surfaces
  via the normal pool, fixed ``Activation_Score = 50.0``
- ``archived``: previously dynamic bucket whose score fell below threshold;
  excluded from default search results
"""


VALID_BUCKET_TYPES: frozenset[str] = frozenset(
    ("dynamic", "permanent", "feel", "archived")
)


@dataclass
class MemoryBucket:
    """One memory bucket — a single addressable unit of long-term memory.

    Field ordering matches what is written to the ``memories`` SQLite table.
    Defaults are chosen so that ``MemoryBucket(id=..., session_id=...,
    content=...)`` produces a valid row with a neutral emotional signature.
    """

    id: str
    session_id: str
    content: str
    name: str = ""
    domain: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    valence: float = 0.5
    arousal: float = 0.3
    importance: int = 5
    bucket_type: BucketType = "dynamic"
    pinned: bool = False
    resolved: bool = False
    digested: bool = False
    model_valence: float | None = None
    source_bucket_id: str | None = None
    activation_count: float = 0.0
    created_at: float = 0.0
    last_active_at: float = 0.0


def generate_bucket_id() -> str:
    """Return a fresh 12-char hex bucket id (first 12 hex of UUID4).

    12 hex chars give 48 bits of entropy → ~2.8e14 unique ids, more than
    enough per session even for very chatty users. The short id is friendly
    to log lines and command output.
    """
    return uuid.uuid4().hex[:12]


def _clamp_float(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into [low, high] using min/max (handles NaN as low).

    NaN values comparing False to both bounds would fall through; we coerce
    them to ``low`` to keep downstream math well-defined.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return low
    if v != v:  # NaN check without importing math
        return low
    if v < low:
        return low
    if v > high:
        return high
    return v


def _clamp_int(value: int, low: int, high: int) -> int:
    """Clamp an integer value to the inclusive [low, high] range."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return low
    if v < low:
        return low
    if v > high:
        return high
    return v


def clamp_bucket(bucket: MemoryBucket) -> MemoryBucket:
    """Clamp a bucket's numeric fields to their valid ranges in place.

    Returns the same instance for convenient chaining. This is the single
    source of truth for input sanitisation: every write path (LLM tool,
    /memory command, Dashboard PATCH) must funnel through it.
    """
    bucket.valence = _clamp_float(bucket.valence, 0.0, 1.0)
    bucket.arousal = _clamp_float(bucket.arousal, 0.0, 1.0)
    bucket.importance = _clamp_int(bucket.importance, 1, 10)

    if bucket.model_valence is not None:
        bucket.model_valence = _clamp_float(bucket.model_valence, 0.0, 1.0)

    # ``activation_count`` is a float; floor it at 0 to avoid negatives.
    if bucket.activation_count < 0:
        bucket.activation_count = 0.0

    # Pinned buckets are by definition core principles → lock importance
    # and bucket_type so the rest of the pipeline can rely on the invariant.
    if bucket.pinned:
        bucket.importance = 10
        if bucket.bucket_type == "dynamic":
            bucket.bucket_type = "permanent"

    if bucket.bucket_type not in VALID_BUCKET_TYPES:
        bucket.bucket_type = "dynamic"

    return bucket


def new_bucket(
    session_id: str,
    content: str,
    *,
    name: str = "",
    domain: list[str] | None = None,
    tags: list[str] | None = None,
    valence: float = 0.5,
    arousal: float = 0.3,
    importance: int = 5,
    bucket_type: BucketType = "dynamic",
    pinned: bool = False,
) -> MemoryBucket:
    """Convenience factory that produces a new, ready-to-persist bucket.

    Sets ``id`` via ``generate_bucket_id``, stamps both timestamps to now,
    initialises ``activation_count`` to 0 (per Requirement 2.7) and applies
    ``clamp_bucket`` so the caller can pass user-supplied values without
    pre-validation.
    """
    now = time.time()
    bucket = MemoryBucket(
        id=generate_bucket_id(),
        session_id=session_id,
        content=content,
        name=name,
        domain=list(domain or []),
        tags=list(tags or []),
        valence=valence,
        arousal=arousal,
        importance=importance,
        bucket_type=bucket_type,
        pinned=pinned,
        resolved=False,
        digested=False,
        model_valence=None,
        source_bucket_id=None,
        activation_count=0.0,
        created_at=now,
        last_active_at=now,
    )
    return clamp_bucket(bucket)
