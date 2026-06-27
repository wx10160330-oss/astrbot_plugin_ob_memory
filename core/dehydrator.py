"""Memory dehydration: LLM-powered content compression for injection.

When memories are injected into the system_prompt, long content needs to
be compressed into high-density summaries to save tokens while preserving
all key facts. This module provides:

- :func:`dehydrate` — compress a single memory's content via LLM
- In-memory cache keyed by content hash to avoid redundant API calls

The dehydrator is optional: when no LLM provider is available or the call
fails, it falls back to returning the full content (no truncation).
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tagger import Tagger

logger = logging.getLogger("astrbot_plugin_ob_memory.dehydrator")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEHYDRATE_THRESHOLD: int = 200
"""Content shorter than this (in chars) is returned as-is."""

CACHE_MAX_SIZE: int = 256
"""LRU cache capacity. Each entry is ~100-200 chars of summary text."""

DEHYDRATE_SYSTEM_PROMPT: str = """\
你是一个信息压缩专家。将以下记忆内容压缩为一段紧凑摘要（不超过150字）。

压缩规则：
1. 保留所有核心事实（人物、时间、地点、数字、决定）
2. 保留当前情绪状态和态度
3. 保留所有待办/未完成事项
4. 去除冗余修饰、重复表述、口水话
5. 使用原文的人称视角（第一人称/第二人称保持不变）

直接输出压缩后的文本，不要加任何前缀、标题或格式标记。"""


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------
class _LRUCache:
    """Simple LRU cache backed by OrderedDict."""

    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self._store: OrderedDict[str, str] = OrderedDict()
        self._max = max_size

    def get(self, key: str) -> str | None:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self._max:
                self._store.popitem(last=False)
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Dehydrator
# ---------------------------------------------------------------------------
class Dehydrator:
    """Compress long memory content into high-density summaries via LLM.

    Usage::

        dehydrator = Dehydrator(tagger)
        summary = await dehydrator.dehydrate(bucket.content)
        # Returns compressed text, or original if short / LLM unavailable
    """

    def __init__(self, tagger: Tagger | None = None):
        self.tagger = tagger
        self._cache = _LRUCache(CACHE_MAX_SIZE)

    async def dehydrate(self, content: str, *, session_id: str | None = None) -> str:
        """Compress ``content`` if it exceeds the threshold.

        Returns the compressed summary on success, or the full original
        content if:
        - Content is short enough (< DEHYDRATE_THRESHOLD chars)
        - No tagger/LLM provider is available
        - The LLM call fails

        Never truncates. The caller always gets usable text.
        """
        if not content:
            return ""

        text = content.strip().replace("\n", " ")

        # Short content: return as-is
        if len(text) <= DEHYDRATE_THRESHOLD:
            return text

        # Check cache
        cache_key = _content_hash(text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # No tagger: return full content (no truncation)
        if self.tagger is None:
            return text

        # Call LLM to compress
        try:
            compressed = await self.tagger._call(
                session_id,
                system_prompt=DEHYDRATE_SYSTEM_PROMPT,
                user_prompt=text,
            )
        except Exception as e:
            logger.debug("dehydrate LLM call failed: %s", e)
            return text

        if not compressed or len(compressed) >= len(text):
            # LLM returned empty or longer text — use original
            return text

        # Cache and return
        self._cache.put(cache_key, compressed)
        return compressed


def _content_hash(text: str) -> str:
    """Fast hash for cache key. MD5 is fine here (not security-sensitive)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
