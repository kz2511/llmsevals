"""Tests for Hallucination, Toxicity, Bias, and Coherence metrics (mocked judge)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from llmsevals.core.test_case import TestCase
from llmsevals.metrics.hallucination import Hallucination
from llmsevals.metrics.toxicity import Toxicity
from llmsevals.metrics.bias import Bias
from llmsevals.metrics.coherence import Coherence


# =============================================================================
# Hallucination Tests
# =============================================================================


@pytest.mark.asyncio
async def test_hallucination_no_output():
    """Should return score 0 when actual_output is missing."""
    metric = Hallucination()
    tc = TestCase(input="Question", context=["Some context"])
    result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "No actual output" in result.reason


@pytest.mark.asyncio
async def test_hallucination_no_context():
    """Should return score 0 when context is missing."""
    metric = Hallucination()
    tc = TestCase(input="Question", actual_output="Answer")
    result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "No context provided" in result.reason
    assert result.metadata.get("warning") == "missing_context"


@pytest.mark.asyncio
async def test_hallucination_success():
    """Should return judge score on successful evaluation."""
    metric = Hallucination(threshold=0.7)

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        return_value={"score": 0.9, "reason": "Mostly factual"},
    ):
        tc = TestCase(
            input="Q",
            actual_output="Python was created in 1991.",
            context=["Python was first released in 1991."],
        )
        result = await metric.evaluate(tc)

    assert result.score == 0.9
    assert result.passed is True
    assert result.name == "Hallucination"
    assert result.metadata["context_chunks"] == 1


@pytest.mark.asyncio
async def test_hallucination_judge_error():
    """Should handle judge errors gracefully."""
    metric = Hallucination()

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        side_effect=RuntimeError("Judge crashed"),
    ):
        tc = TestCase(input="Q", actual_output="A", context=["C"])
        result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "Evaluation error" in result.reason


@pytest.mark.asyncio
async def test_hallucination_name():
    """Should have correct metric name."""
    assert Hallucination().name == "Hallucination"


@pytest.mark.asyncio
async def test_hallucination_default_threshold():
    """Default threshold should be 0.7."""
    assert Hallucination().threshold == 0.7


# =============================================================================
# Toxicity Tests
# =============================================================================


@pytest.mark.asyncio
async def test_toxicity_no_output():
    """Should return score 0 when actual_output is missing."""
    metric = Toxicity()
    tc = TestCase(input="Tell me something")
    result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "No actual output" in result.reason


@pytest.mark.asyncio
async def test_toxicity_success():
    """Should return judge score on successful evaluation."""
    metric = Toxicity(threshold=0.8)

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        return_value={"score": 1.0, "reason": "Completely safe content"},
    ):
        tc = TestCase(input="Hi", actual_output="Hello, how can I help?")
        result = await metric.evaluate(tc)

    assert result.score == 1.0
    assert result.passed is True
    assert result.name == "Toxicity"


@pytest.mark.asyncio
async def test_toxicity_low_score():
    """Should fail when content is toxic."""
    metric = Toxicity(threshold=0.8)

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        return_value={"score": 0.3, "reason": "Contains offensive language"},
    ):
        tc = TestCase(input="Hi", actual_output="Bad content here")
        result = await metric.evaluate(tc)

    assert result.score == 0.3
    assert result.passed is False


@pytest.mark.asyncio
async def test_toxicity_judge_error():
    """Should handle judge errors gracefully."""
    metric = Toxicity()

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        side_effect=Exception("API error"),
    ):
        tc = TestCase(input="Hi", actual_output="Hello")
        result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "Evaluation error" in result.reason


@pytest.mark.asyncio
async def test_toxicity_default_threshold():
    """Default threshold should be 0.8."""
    assert Toxicity().threshold == 0.8


# =============================================================================
# Bias Tests
# =============================================================================


@pytest.mark.asyncio
async def test_bias_no_output():
    """Should return score 0 when actual_output is missing."""
    metric = Bias()
    tc = TestCase(input="Describe a person")
    result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "No actual output" in result.reason


@pytest.mark.asyncio
async def test_bias_success():
    """Should return judge score on successful evaluation."""
    metric = Bias(threshold=0.7)

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        return_value={"score": 0.95, "reason": "Neutral and inclusive"},
    ):
        tc = TestCase(input="Q", actual_output="A balanced answer.")
        result = await metric.evaluate(tc)

    assert result.score == 0.95
    assert result.passed is True
    assert result.name == "Bias"


@pytest.mark.asyncio
async def test_bias_judge_error():
    """Should handle judge errors gracefully."""
    metric = Bias()

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        side_effect=RuntimeError("Crashed"),
    ):
        tc = TestCase(input="Q", actual_output="A")
        result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "Evaluation error" in result.reason


@pytest.mark.asyncio
async def test_bias_default_threshold():
    """Default threshold should be 0.7."""
    assert Bias().threshold == 0.7


# =============================================================================
# Coherence Tests
# =============================================================================


@pytest.mark.asyncio
async def test_coherence_no_output():
    """Should return score 0 when actual_output is missing."""
    metric = Coherence()
    tc = TestCase(input="Explain something")
    result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "No actual output" in result.reason


@pytest.mark.asyncio
async def test_coherence_success():
    """Should return judge score on successful evaluation."""
    metric = Coherence(threshold=0.5)

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        return_value={"score": 0.85, "reason": "Well-structured text"},
    ):
        tc = TestCase(input="Q", actual_output="A well-structured answer.")
        result = await metric.evaluate(tc)

    assert result.score == 0.85
    assert result.passed is True
    assert result.name == "Coherence"


@pytest.mark.asyncio
async def test_coherence_judge_error():
    """Should handle judge errors gracefully."""
    metric = Coherence()

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        side_effect=Exception("Failed"),
    ):
        tc = TestCase(input="Q", actual_output="A")
        result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "Evaluation error" in result.reason


@pytest.mark.asyncio
async def test_coherence_default_threshold():
    """Default threshold should be 0.5."""
    assert Coherence().threshold == 0.5
