"""Tests for AnswerRelevancy and Faithfulness metrics (mocked judge)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from llmsevals.core.test_case import TestCase
from llmsevals.metrics.answer_relevancy import AnswerRelevancy
from llmsevals.metrics.faithfulness import Faithfulness


# =============================================================================
# AnswerRelevancy Tests
# =============================================================================


@pytest.mark.asyncio
async def test_answer_relevancy_no_output():
    """Should return score 0 when actual_output is missing."""
    metric = AnswerRelevancy()
    tc = TestCase(input="What is AI?")
    result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert result.passed is False
    assert "No actual output" in result.reason


@pytest.mark.asyncio
async def test_answer_relevancy_success():
    """Should return judge score on successful evaluation."""
    metric = AnswerRelevancy(threshold=0.5)

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        return_value={"score": 0.85, "reason": "Highly relevant answer"},
    ):
        tc = TestCase(
            input="What is Python?",
            actual_output="Python is a programming language.",
        )
        result = await metric.evaluate(tc)

    assert result.score == 0.85
    assert result.passed is True
    assert result.reason == "Highly relevant answer"
    assert result.name == "Answer Relevancy"


@pytest.mark.asyncio
async def test_answer_relevancy_low_score():
    """Should fail when score is below threshold."""
    metric = AnswerRelevancy(threshold=0.8)

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        return_value={"score": 0.3, "reason": "Off topic"},
    ):
        tc = TestCase(
            input="What is Python?",
            actual_output="The sky is blue.",
        )
        result = await metric.evaluate(tc)

    assert result.score == 0.3
    assert result.passed is False


@pytest.mark.asyncio
async def test_answer_relevancy_judge_error():
    """Should handle judge evaluation errors gracefully."""
    metric = AnswerRelevancy()

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        side_effect=Exception("API connection failed"),
    ):
        tc = TestCase(
            input="What is AI?",
            actual_output="AI is artificial intelligence.",
        )
        result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert result.passed is False
    assert "Evaluation error" in result.reason


@pytest.mark.asyncio
async def test_answer_relevancy_name():
    """Should have correct metric name."""
    metric = AnswerRelevancy()
    assert metric.name == "Answer Relevancy"


# =============================================================================
# Faithfulness Tests
# =============================================================================


@pytest.mark.asyncio
async def test_faithfulness_no_output():
    """Should return score 0 when actual_output is missing."""
    metric = Faithfulness()
    tc = TestCase(input="Question", context=["Some context"])
    result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "No actual output" in result.reason


@pytest.mark.asyncio
async def test_faithfulness_no_context():
    """Should return score 0 when context is missing."""
    metric = Faithfulness()
    tc = TestCase(input="Question", actual_output="Answer")
    result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "No context provided" in result.reason
    assert result.metadata.get("warning") == "missing_context"


@pytest.mark.asyncio
async def test_faithfulness_success():
    """Should return judge score on successful evaluation."""
    metric = Faithfulness(threshold=0.7)

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        return_value={"score": 0.95, "reason": "Fully faithful to context"},
    ):
        tc = TestCase(
            input="When was Python created?",
            actual_output="Python was created in 1991.",
            context=["Python was first released in 1991."],
        )
        result = await metric.evaluate(tc)

    assert result.score == 0.95
    assert result.passed is True
    assert result.metadata["context_chunks"] == 1
    assert result.name == "Faithfulness"


@pytest.mark.asyncio
async def test_faithfulness_multiple_context_chunks():
    """Should join multiple context chunks."""
    metric = Faithfulness()

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        return_value={"score": 0.8, "reason": "Mostly faithful"},
    ) as mock_eval:
        tc = TestCase(
            input="Q",
            actual_output="A",
            context=["Chunk 1", "Chunk 2", "Chunk 3"],
        )
        result = await metric.evaluate(tc)

        # Verify context was joined
        call_args = mock_eval.call_args
        context_sent = call_args.kwargs["variables"]["context"]
        assert "Chunk 1" in context_sent
        assert "Chunk 2" in context_sent
        assert "Chunk 3" in context_sent
        assert result.metadata["context_chunks"] == 3


@pytest.mark.asyncio
async def test_faithfulness_low_score():
    """Should fail when score is below threshold."""
    metric = Faithfulness(threshold=0.8)

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        return_value={"score": 0.4, "reason": "Many hallucinated claims"},
    ):
        tc = TestCase(
            input="Q",
            actual_output="A",
            context=["Context"],
        )
        result = await metric.evaluate(tc)

    assert result.score == 0.4
    assert result.passed is False


@pytest.mark.asyncio
async def test_faithfulness_judge_error():
    """Should handle judge errors gracefully."""
    metric = Faithfulness()

    with patch.object(
        metric._judge, "evaluate", new_callable=AsyncMock,
        side_effect=RuntimeError("Judge crashed"),
    ):
        tc = TestCase(
            input="Q",
            actual_output="A",
            context=["Context"],
        )
        result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert "Evaluation error" in result.reason


@pytest.mark.asyncio
async def test_faithfulness_name():
    """Should have correct metric name."""
    metric = Faithfulness()
    assert metric.name == "Faithfulness"


@pytest.mark.asyncio
async def test_faithfulness_default_threshold():
    """Default threshold should be 0.7."""
    metric = Faithfulness()
    assert metric.threshold == 0.7
