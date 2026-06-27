"""Unit tests for ``core.models`` clamping and id generation."""

from __future__ import annotations

import math
import re

import pytest

from astrbot_plugin_ob_memory.core.models import (
    MemoryBucket,
    clamp_bucket,
    generate_bucket_id,
    new_bucket,
)


def test_generate_bucket_id_format():
    bid = generate_bucket_id()
    assert isinstance(bid, str)
    assert len(bid) == 12
    assert re.fullmatch(r"[0-9a-f]{12}", bid) is not None


def test_generate_bucket_id_uniqueness_basic():
    ids = {generate_bucket_id() for _ in range(1000)}
    # Collisions on 12 hex chars over 1000 draws are astronomically unlikely.
    assert len(ids) == 1000


def test_clamp_valence_arousal_within_range():
    b = MemoryBucket(id="x", session_id="s", content="c", valence=1.5, arousal=-0.4)
    clamp_bucket(b)
    assert b.valence == 1.0
    assert b.arousal == 0.0


def test_clamp_valence_handles_nan():
    b = MemoryBucket(id="x", session_id="s", content="c", valence=math.nan)
    clamp_bucket(b)
    assert b.valence == 0.0


def test_clamp_importance_to_inclusive_range():
    b = MemoryBucket(id="x", session_id="s", content="c", importance=99)
    clamp_bucket(b)
    assert b.importance == 10
    b.importance = -3
    clamp_bucket(b)
    assert b.importance == 1


def test_pinned_locks_importance_and_type():
    b = MemoryBucket(
        id="x", session_id="s", content="c",
        importance=4, pinned=True, bucket_type="dynamic",
    )
    clamp_bucket(b)
    assert b.importance == 10
    assert b.bucket_type == "permanent"


def test_pinned_does_not_demote_feel_bucket():
    # A feel bucket that is also pinned should keep its feel type, since
    # feel has priority semantics (model's reflection, never surfaces).
    b = MemoryBucket(
        id="x", session_id="s", content="c",
        importance=5, pinned=True, bucket_type="feel",
    )
    clamp_bucket(b)
    assert b.bucket_type == "feel"
    assert b.importance == 10


def test_invalid_bucket_type_falls_back_to_dynamic():
    b = MemoryBucket(id="x", session_id="s", content="c", bucket_type="weird")  # type: ignore[arg-type]
    clamp_bucket(b)
    assert b.bucket_type == "dynamic"


def test_negative_activation_count_is_floored():
    b = MemoryBucket(id="x", session_id="s", content="c", activation_count=-2.0)
    clamp_bucket(b)
    assert b.activation_count == 0.0


def test_new_bucket_factory_initialises_timestamps():
    b = new_bucket("session-A", "hello world", importance=8)
    assert b.session_id == "session-A"
    assert b.content == "hello world"
    assert b.importance == 8
    assert b.activation_count == 0.0
    assert b.created_at > 0
    assert b.last_active_at == b.created_at
    assert b.bucket_type == "dynamic"


def test_new_bucket_clamps_user_supplied_values():
    b = new_bucket("s", "c", valence=2.5, arousal=-0.1, importance=99)
    assert 0.0 <= b.valence <= 1.0
    assert 0.0 <= b.arousal <= 1.0
    assert 1 <= b.importance <= 10


@pytest.mark.parametrize("bad", ["nope", "feels", ""])
def test_invalid_string_does_not_explode(bad):
    b = MemoryBucket(id=bad, session_id="s", content="c")
    clamp_bucket(b)
    assert b.id == bad  # id is opaque; clamp shouldn't mangle it
