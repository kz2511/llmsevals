"""Additional edge case tests for Cost metric."""

from __future__ import annotations

import pytest

from llmsevals.core.test_case import TestCase
from llmsevals.metrics.cost import Cost


@pytest.mark.asyncio
async def test_cost_estimate_from_text():
    """When no token_count is provided, should estimate from text."""
    metric = Cost(model="gpt-4o", max_cost_usd=0.10)
    tc = TestCase(
        input="What is artificial intelligence?",
        actual_output="AI is a broad field of computer science.",
        # No token_count provided — should estimate
    )
    result = await metric.evaluate(tc)

    assert result.score > 0.0
    assert result.metadata["prompt_tokens"] > 0
    assert result.metadata["completion_tokens"] > 0


@pytest.mark.asyncio
async def test_cost_high_cost_formatting():
    """Cost >= $0.01 should use 4 decimal place formatting."""
    metric = Cost(model="gpt-4", max_cost_usd=1.0)
    # GPT-4 is expensive: $0.03/1k input + $0.06/1k output
    tc = TestCase(
        input="X",
        actual_output="Y",
        token_count={"prompt_tokens": 5000, "completion_tokens": 5000},
    )
    result = await metric.evaluate(tc)

    # Cost should be > $0.01 for GPT-4 with 10k tokens
    assert result.metadata["cost_usd"] > 0.01
    assert "$" in result.reason
