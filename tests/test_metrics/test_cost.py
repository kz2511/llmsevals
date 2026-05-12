"""Tests for the Cost metric."""

from __future__ import annotations

import pytest

from llmsevals.core.test_case import TestCase
from llmsevals.metrics.cost import Cost


@pytest.mark.asyncio
async def test_cost_with_token_counts():
    """Should calculate cost correctly with token count data."""
    metric = Cost(model="gpt-4o", max_cost_usd=0.10)
    tc = TestCase(
        input="What is AI?",
        actual_output="AI is artificial intelligence.",
        token_count={
            "prompt_tokens": 100,
            "completion_tokens": 200,
        },
    )
    result = await metric.evaluate(tc)

    assert result.score > 0.0
    assert result.passed is True
    assert result.name == "Cost"
    assert "cost_usd" in result.metadata
    assert result.metadata["cost_usd"] > 0


@pytest.mark.asyncio
async def test_cost_cheap_model():
    """Cheap model should score very high."""
    metric = Cost(model="gpt-4o-mini", max_cost_usd=0.10)
    tc = TestCase(
        input="Hello",
        actual_output="Hi",
        token_count={"prompt_tokens": 5, "completion_tokens": 5},
    )
    result = await metric.evaluate(tc)

    assert result.score > 0.99  # Very cheap
    assert result.passed is True


@pytest.mark.asyncio
async def test_cost_expensive_call():
    """Expensive call should score low."""
    metric = Cost(model="gpt-4", max_cost_usd=0.01)
    tc = TestCase(
        input="Write a long essay",
        actual_output="A" * 5000,
        token_count={"prompt_tokens": 500, "completion_tokens": 2000},
    )
    result = await metric.evaluate(tc)

    assert result.score == 0.0  # Over budget
    assert result.passed is False


@pytest.mark.asyncio
async def test_cost_no_model():
    """Missing model should return score 0 with warning."""
    metric = Cost()  # No model specified
    tc = TestCase(
        input="test",
        actual_output="response",
    )
    result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "missing_model" in result.metadata.get("warning", "")


@pytest.mark.asyncio
async def test_cost_unknown_model():
    """Unknown model should return neutral score."""
    metric = Cost(model="totally-unknown-model-xyz")
    tc = TestCase(
        input="test",
        actual_output="response",
        token_count={"prompt_tokens": 100, "completion_tokens": 100},
    )
    result = await metric.evaluate(tc)

    assert result.score == 0.5
    assert "unknown_pricing" in result.metadata.get("warning", "")


@pytest.mark.asyncio
async def test_cost_model_from_metadata():
    """Should read model from test_case.metadata if not set in metric."""
    metric = Cost(max_cost_usd=0.10)
    tc = TestCase(
        input="test",
        actual_output="response",
        token_count={"prompt_tokens": 50, "completion_tokens": 50},
        metadata={"model": "gpt-4o-mini"},
    )
    result = await metric.evaluate(tc)

    assert result.score > 0.0
    assert result.metadata["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_cost_metadata_contents():
    """Metadata should contain detailed cost breakdown."""
    metric = Cost(model="gpt-4o", max_cost_usd=0.10)
    tc = TestCase(
        input="test",
        actual_output="response",
        token_count={"prompt_tokens": 100, "completion_tokens": 200},
    )
    result = await metric.evaluate(tc)

    assert "cost_usd" in result.metadata
    assert "prompt_tokens" in result.metadata
    assert "completion_tokens" in result.metadata
    assert "total_tokens" in result.metadata
    assert result.metadata["total_tokens"] == 300
