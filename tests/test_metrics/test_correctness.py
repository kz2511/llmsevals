"""Tests for AnswerCorrectness metric."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llmsevals.core.test_case import TestCase
from llmsevals.metrics.correctness import AnswerCorrectness


def _tc(actual=None, expected=None):
    return TestCase(
        input="When was Python created?",
        actual_output=actual,
        expected_output=expected,
    )


def test_correctness_name():
    assert AnswerCorrectness().name == "Answer Correctness"


def test_correctness_default_threshold():
    assert AnswerCorrectness().threshold == 0.7


def test_correctness_required_fields():
    assert "actual_output" in AnswerCorrectness().required_fields
    assert "expected_output" in AnswerCorrectness().required_fields


@pytest.mark.asyncio
async def test_correctness_no_actual_output():
    result = await AnswerCorrectness().evaluate(_tc(expected="1991"))
    assert result.score == 0.0
    assert "No actual output" in result.reason


@pytest.mark.asyncio
async def test_correctness_no_expected_output():
    result = await AnswerCorrectness().evaluate(_tc(actual="1991"))
    assert result.score == 0.0
    assert "expected_output" in result.reason
    assert result.metadata.get("warning") == "missing_expected_output"


@pytest.mark.asyncio
async def test_correctness_high_score_passes():
    metric = AnswerCorrectness(threshold=0.7)
    with patch.object(
        metric._judge, "evaluate", new=AsyncMock(return_value={"score": 0.9, "reason": "Correct"})
    ):
        result = await metric.evaluate(_tc(actual="1991", expected="Python was released in 1991."))
    assert result.score == 0.9
    assert result.passed is True


@pytest.mark.asyncio
async def test_correctness_low_score_fails():
    metric = AnswerCorrectness(threshold=0.7)
    with patch.object(
        metric._judge, "evaluate", new=AsyncMock(return_value={"score": 0.3, "reason": "Wrong"})
    ):
        result = await metric.evaluate(_tc(actual="1995", expected="1991"))
    assert result.score == 0.3
    assert result.passed is False


@pytest.mark.asyncio
async def test_correctness_judge_error_returns_zero():
    metric = AnswerCorrectness()
    with patch.object(metric._judge, "evaluate", side_effect=Exception("API down")):
        result = await metric.evaluate(_tc(actual="1991", expected="1991"))
    assert result.score == 0.0
    assert "Evaluation error" in result.reason


@pytest.mark.asyncio
async def test_correctness_importable_from_metrics():
    from llmsevals import metrics
    assert hasattr(metrics, "AnswerCorrectness")
    assert metrics.AnswerCorrectness().name == "Answer Correctness"
