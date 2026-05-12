"""Tests for the BaseMetric abstract class."""

from __future__ import annotations

import pytest

from llmsevals.core.base_metric import BaseMetric, MetricResult


# Concrete implementation for testing
class DummyMetric(BaseMetric):
    """A simple metric for testing."""

    @property
    def name(self) -> str:
        return "Dummy"

    async def evaluate(self, test_case) -> MetricResult:
        score = 0.75
        return self._make_result(score=score, reason="Test result")


def test_metric_result_creation():
    """Should create MetricResult with all fields."""
    mr = MetricResult(
        name="Test",
        score=0.85,
        passed=True,
        reason="Good score",
        metadata={"key": "value"},
    )
    assert mr.name == "Test"
    assert mr.score == 0.85
    assert mr.passed is True
    assert mr.reason == "Good score"
    assert mr.metadata["key"] == "value"


def test_metric_result_score_bounds():
    """Score should be bounded between 0.0 and 1.0."""
    with pytest.raises(Exception):
        MetricResult(name="T", score=1.5, passed=True)

    with pytest.raises(Exception):
        MetricResult(name="T", score=-0.1, passed=True)


def test_metric_result_repr():
    """Should show pass/fail icon in repr."""
    mr_pass = MetricResult(name="Test", score=0.8, passed=True, reason="OK")
    assert "✅" in repr(mr_pass)

    mr_fail = MetricResult(name="Test", score=0.2, passed=False, reason="Bad")
    assert "❌" in repr(mr_fail)


def test_base_metric_threshold_validation():
    """Threshold should be between 0 and 1."""
    with pytest.raises(ValueError, match="Threshold must be"):
        DummyMetric(threshold=1.5)

    with pytest.raises(ValueError, match="Threshold must be"):
        DummyMetric(threshold=-0.1)


def test_base_metric_default_threshold():
    """Default threshold should be 0.5."""
    m = DummyMetric()
    assert m.threshold == 0.5


def test_base_metric_custom_threshold():
    """Custom threshold should be stored."""
    m = DummyMetric(threshold=0.8)
    assert m.threshold == 0.8


@pytest.mark.asyncio
async def test_base_metric_make_result_pass():
    """_make_result should auto-detect pass when score >= threshold."""
    m = DummyMetric(threshold=0.5)
    result = m._make_result(score=0.7, reason="Above threshold")

    assert result.passed is True
    assert result.score == 0.7
    assert result.name == "Dummy"


@pytest.mark.asyncio
async def test_base_metric_make_result_fail():
    """_make_result should auto-detect fail when score < threshold."""
    m = DummyMetric(threshold=0.8)
    result = m._make_result(score=0.5, reason="Below threshold")

    assert result.passed is False


@pytest.mark.asyncio
async def test_base_metric_make_result_with_metadata():
    """_make_result should include metadata."""
    m = DummyMetric()
    result = m._make_result(
        score=0.9,
        reason="Good",
        metadata={"detail": "extra"},
    )
    assert result.metadata["detail"] == "extra"


def test_base_metric_repr():
    """__repr__ should show class name and threshold."""
    m = DummyMetric(threshold=0.7)
    assert "DummyMetric" in repr(m)
    assert "0.7" in repr(m)
