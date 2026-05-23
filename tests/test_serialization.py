"""Round-trip tests for ``bucket_to_row`` / ``row_to_bucket``."""

from __future__ import annotations

import time

import pytest

from astrbot_plugin_ob_memory.core.memory_manager import bucket_to_row, row_to_bucket
from astrbot_plugin_ob_memory.core.models import MemoryBucket, new_bucket


def _row_dict_from_tuple(row_tuple, columns):
    """Tiny helper: build a dict-like object that supports ``row[key]``.

    Mirrors aiosqlite.Row well enough for the conversion logic.
    """

    class FakeRow(dict):
        def __getitem__(self, key):
            if isinstance(key, str):
                return super().__getitem__(key)
            return list(self.values())[key]

    return FakeRow(zip(columns, row_tuple, strict=True))


COLUMNS = (
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


def test_round_trip_basic():
    b = new_bucket(
        "session-A",
        "I got the offer",
        name="实习offer",
        tags=["实习", "offer"],
        domain=["成长", "求职"],
        valence=0.8,
        arousal=0.7,
        importance=7,
    )
    row = _row_dict_from_tuple(bucket_to_row(b), COLUMNS)
    restored = row_to_bucket(row)

    assert restored.id == b.id
    assert restored.session_id == b.session_id
    assert restored.content == b.content
    assert restored.tags == b.tags
    assert restored.domain == b.domain
    assert restored.valence == pytest.approx(b.valence)
    assert restored.arousal == pytest.approx(b.arousal)
    assert restored.importance == b.importance
    assert restored.bucket_type == b.bucket_type
    assert restored.pinned == b.pinned
    assert restored.resolved == b.resolved
    assert restored.digested == b.digested
    assert restored.activation_count == pytest.approx(b.activation_count)


def test_round_trip_preserves_none_columns():
    b = MemoryBucket(
        id="x", session_id="s", content="c",
        model_valence=None, source_bucket_id=None,
        created_at=time.time(), last_active_at=time.time(),
    )
    row = _row_dict_from_tuple(bucket_to_row(b), COLUMNS)
    restored = row_to_bucket(row)
    assert restored.model_valence is None
    assert restored.source_bucket_id is None


def test_round_trip_unicode_safe():
    b = new_bucket(
        "群:1234",
        "今天有点累",
        name="情绪低落",
        tags=["疲惫", "情感"],
        domain=["内心"],
    )
    row = _row_dict_from_tuple(bucket_to_row(b), COLUMNS)
    restored = row_to_bucket(row)
    assert restored.name == "情绪低落"
    assert "疲惫" in restored.tags
    assert restored.domain == ["内心"]
