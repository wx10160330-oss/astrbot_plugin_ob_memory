"""Unit tests for ``core.tagger``.

Uses a stub LLM provider that returns canned ``LLMResponse``-shaped
objects. We focus on:

- analyze: correct field extraction + clamping + fallback
- merge_content: returns merged text, falls back to concatenation on
  failure
- judge_worth_recording: returns (bool, str) and stays conservative on
  errors
- robustness: code fences, trailing prose, malformed JSON
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from astrbot_plugin_ob_memory.core.tagger import DEFAULT_ANALYZE, Tagger

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
@dataclass
class FakeLLMResponse:
    completion_text: str


class FakeProvider:
    """Returns one of a queue of canned responses, then loops on the last."""

    def __init__(self, responses: list[str]):
        self._queue = list(responses)
        self.calls: list[tuple[str, str]] = []  # (system, user)

    async def text_chat(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> FakeLLMResponse:
        self.calls.append((system_prompt or "", prompt or ""))
        if self._queue:
            text = self._queue.pop(0)
        else:
            text = ""
        return FakeLLMResponse(completion_text=text)


class ExplodingProvider:
    async def text_chat(self, **kwargs) -> FakeLLMResponse:
        raise RuntimeError("LLM down")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
async def test_analyze_returns_default_without_provider():
    tagger = Tagger(context=None)
    result = await tagger.analyze("anything")
    assert result == DEFAULT_ANALYZE


async def test_analyze_returns_default_for_empty_content():
    provider = FakeProvider(["{}"])
    tagger = Tagger(context=None, fixed_provider=provider)
    result = await tagger.analyze("   ")
    assert result == DEFAULT_ANALYZE
    # Must not have called the LLM for empty content.
    assert provider.calls == []


async def test_analyze_parses_clean_json():
    response = (
        '{"domain": ["成长", "求职"], "valence": 0.8, "arousal": 0.7, '
        '"tags": ["实习", "offer"], "suggested_name": "实习offer", '
        '"importance": 7}'
    )
    provider = FakeProvider([response])
    tagger = Tagger(context=None, fixed_provider=provider)
    result = await tagger.analyze("拿到实习 offer")
    assert result["domain"] == ["成长", "求职"]
    assert result["valence"] == 0.8
    assert result["arousal"] == 0.7
    assert result["tags"] == ["实习", "offer"]
    assert result["suggested_name"] == "实习offer"
    assert result["importance"] == 7


async def test_analyze_strips_code_fences():
    response = (
        "```json\n"
        '{"domain": ["内心"], "valence": 0.3, "arousal": 0.6, '
        '"tags": ["焦虑"], "suggested_name": "焦虑", "importance": 6}\n'
        "```"
    )
    provider = FakeProvider([response])
    tagger = Tagger(context=None, fixed_provider=provider)
    result = await tagger.analyze("最近有点焦虑")
    assert result["domain"] == ["内心"]
    assert result["importance"] == 6


async def test_analyze_handles_trailing_prose():
    response = (
        "Sure! Here's the analysis:\n"
        '{"domain": ["日常"], "valence": 0.5, "arousal": 0.3, "tags": [], '
        '"suggested_name": "", "importance": 4}\n'
        "Hope this helps!"
    )
    provider = FakeProvider([response])
    tagger = Tagger(context=None, fixed_provider=provider)
    result = await tagger.analyze("吃了顿火锅")
    assert result["domain"] == ["日常"]
    assert result["importance"] == 4


async def test_analyze_clamps_out_of_range_values():
    response = (
        '{"domain": ["x"], "valence": 9.9, "arousal": -3, '
        '"tags": [], "suggested_name": "", "importance": 99}'
    )
    provider = FakeProvider([response])
    tagger = Tagger(context=None, fixed_provider=provider)
    result = await tagger.analyze("x")
    assert result["valence"] == 1.0
    assert result["arousal"] == 0.0
    assert result["importance"] == 10


async def test_analyze_falls_back_on_malformed_json():
    provider = FakeProvider(["not json at all{{{"])
    tagger = Tagger(context=None, fixed_provider=provider)
    result = await tagger.analyze("x")
    assert result == DEFAULT_ANALYZE


async def test_analyze_falls_back_when_provider_raises():
    tagger = Tagger(context=None, fixed_provider=ExplodingProvider())
    result = await tagger.analyze("x")
    assert result == DEFAULT_ANALYZE


async def test_analyze_filters_non_string_tags():
    # Some models occasionally emit numbers in tag arrays.
    response = (
        '{"domain": ["x"], "valence": 0.5, "arousal": 0.3, '
        '"tags": ["valid", 42, null, "also-valid"], '
        '"suggested_name": "", "importance": 5}'
    )
    provider = FakeProvider([response])
    tagger = Tagger(context=None, fixed_provider=provider)
    result = await tagger.analyze("x")
    assert result["tags"] == ["valid", "also-valid"]


# ---------------------------------------------------------------------------
# merge_content
# ---------------------------------------------------------------------------
async def test_merge_returns_llm_output():
    provider = FakeProvider(["合并后的统一描述"])
    tagger = Tagger(context=None, fixed_provider=provider)
    merged = await tagger.merge_content("旧内容", "新内容")
    assert merged == "合并后的统一描述"


async def test_merge_falls_back_to_concat_on_empty_response():
    provider = FakeProvider([""])
    tagger = Tagger(context=None, fixed_provider=provider)
    merged = await tagger.merge_content("旧内容", "新内容")
    assert "旧内容" in merged
    assert "新内容" in merged


async def test_merge_short_circuits_when_one_side_empty():
    tagger = Tagger(context=None, fixed_provider=FakeProvider([]))
    assert await tagger.merge_content("only-old", "") == "only-old"
    assert await tagger.merge_content("", "only-new") == "only-new"


async def test_merge_falls_back_when_provider_raises():
    tagger = Tagger(context=None, fixed_provider=ExplodingProvider())
    merged = await tagger.merge_content("a", "b")
    assert "a" in merged
    assert "b" in merged


# ---------------------------------------------------------------------------
# judge_worth_recording
# ---------------------------------------------------------------------------
async def test_judge_returns_true_on_clear_signal():
    response = '{"remember": true, "reason": "用户分享了重要决定"}'
    provider = FakeProvider([response])
    tagger = Tagger(context=None, fixed_provider=provider)
    should, reason = await tagger.judge_worth_recording("user", "ack")
    assert should is True
    assert "决定" in reason


async def test_judge_returns_false_default():
    response = '{"remember": false, "reason": "闲聊"}'
    provider = FakeProvider([response])
    tagger = Tagger(context=None, fixed_provider=provider)
    should, _ = await tagger.judge_worth_recording("hi", "hello")
    assert should is False


async def test_judge_returns_false_for_empty_messages():
    tagger = Tagger(context=None, fixed_provider=FakeProvider([]))
    should, _ = await tagger.judge_worth_recording("", "anything")
    assert should is False
    should, _ = await tagger.judge_worth_recording("anything", "")
    assert should is False


async def test_judge_conservative_on_failure():
    tagger = Tagger(context=None, fixed_provider=ExplodingProvider())
    should, reason = await tagger.judge_worth_recording("u", "a")
    assert should is False
    assert reason  # non-empty reason for debugging


async def test_judge_conservative_on_malformed_json():
    provider = FakeProvider(["{garbage"])
    tagger = Tagger(context=None, fixed_provider=provider)
    should, _ = await tagger.judge_worth_recording("u", "a")
    assert should is False
