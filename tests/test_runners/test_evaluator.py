"""Tests for the Evaluator runner."""

from __future__ import annotations

import pytest

from llmsevals.core.test_case import TestCase
from llmsevals.core.eval_result import EvalResult
from llmsevals.metrics.latency import Latency
from llmsevals.metrics.cost import Cost
from llmsevals.runners.evaluator import Evaluator


def test_evaluator_sync_run():
    """Evaluator.run() should work synchronously."""
    evaluator = Evaluator(
        metrics=[Latency(max_latency_ms=5000)],
        verbose=False,
    )

    test_cases = [
        TestCase(input="Q1", actual_output="A1", latency_ms=1000.0),
        TestCase(input="Q2", actual_output="A2", latency_ms=2000.0),
    ]

    result = evaluator.run(test_cases=test_cases)

    assert isinstance(result, EvalResult)
    assert result.total_test_cases == 2
    assert len(result.test_case_results) == 2


def test_evaluator_all_pass():
    """All test cases should pass when under thresholds."""
    evaluator = Evaluator(
        metrics=[Latency(threshold=0.3, max_latency_ms=5000)],
        verbose=False,
    )

    test_cases = [
        TestCase(input="Q1", actual_output="A1", latency_ms=500.0),
        TestCase(input="Q2", actual_output="A2", latency_ms=1000.0),
    ]

    result = evaluator.run(test_cases=test_cases)

    assert result.passed_count == 2
    assert result.failed_count == 0
    assert result.pass_rate == 1.0


def test_evaluator_mixed_results():
    """Should correctly count pass/fail with mixed results."""
    evaluator = Evaluator(
        metrics=[Latency(threshold=0.7, max_latency_ms=5000)],
        verbose=False,
    )

    test_cases = [
        TestCase(input="Fast", actual_output="A1", latency_ms=500.0),   # pass
        TestCase(input="Slow", actual_output="A2", latency_ms=4000.0),  # fail
    ]

    result = evaluator.run(test_cases=test_cases)

    assert result.passed_count == 1
    assert result.failed_count == 1
    assert result.pass_rate == 0.5


def test_evaluator_multiple_metrics():
    """Should run multiple metrics on each test case."""
    evaluator = Evaluator(
        metrics=[
            Latency(max_latency_ms=5000),
            Cost(model="gpt-4o-mini", max_cost_usd=0.10),
        ],
        verbose=False,
    )

    test_cases = [
        TestCase(
            input="Q1",
            actual_output="A1",
            latency_ms=1000.0,
            token_count={"prompt_tokens": 10, "completion_tokens": 20},
        ),
    ]

    result = evaluator.run(test_cases=test_cases)

    assert result.total_test_cases == 1
    tcr = result.test_case_results[0]
    assert len(tcr.metric_results) == 2
    metric_names = {mr.name for mr in tcr.metric_results}
    assert "Latency" in metric_names
    assert "Cost" in metric_names


def test_evaluator_no_test_cases():
    """Should raise ValueError when no test cases provided."""
    evaluator = Evaluator(
        metrics=[Latency()],
        verbose=False,
    )

    with pytest.raises(ValueError, match="No test cases"):
        evaluator.run()


def test_evaluator_no_metrics():
    """Should raise ValueError when no metrics configured."""
    evaluator = Evaluator(verbose=False)

    with pytest.raises(ValueError, match="No metrics"):
        evaluator.run(test_cases=[TestCase(input="Q1", actual_output="A1")])


def test_evaluator_add_metric():
    """Fluent API should work for adding metrics."""
    evaluator = Evaluator(verbose=False)
    result = evaluator.add_metric(Latency()).add_metric(Cost(model="gpt-4o"))

    assert result is evaluator  # returns self
    assert len(evaluator.metrics) == 2


def test_evaluator_metadata():
    """Result metadata should contain run information."""
    evaluator = Evaluator(
        metrics=[Latency(max_latency_ms=5000)],
        verbose=False,
    )

    result = evaluator.run(
        test_cases=[TestCase(input="Q1", actual_output="A1", latency_ms=500.0)]
    )

    assert "total_time_ms" in result.metadata
    assert "metrics" in result.metadata
    assert "timestamp" in result.metadata
    assert result.metadata["metrics"] == ["Latency"]


def test_evaluator_metric_summary():
    """Metric summary should aggregate per-metric stats."""
    evaluator = Evaluator(
        metrics=[Latency(threshold=0.5, max_latency_ms=5000)],
        verbose=False,
    )

    test_cases = [
        TestCase(input="Q1", actual_output="A1", latency_ms=500.0),
        TestCase(input="Q2", actual_output="A2", latency_ms=3000.0),
    ]

    result = evaluator.run(test_cases=test_cases)
    summary = result.metric_summary()

    assert "Latency" in summary
    assert "avg_score" in summary["Latency"]
    assert "pass_rate" in summary["Latency"]
    assert "min_score" in summary["Latency"]
    assert "max_score" in summary["Latency"]


def test_evaluator_to_json():
    """Results should serialize to JSON."""
    evaluator = Evaluator(
        metrics=[Latency(max_latency_ms=5000)],
        verbose=False,
    )

    result = evaluator.run(
        test_cases=[TestCase(input="Q1", actual_output="A1", latency_ms=500.0)]
    )

    json_str = result.to_json()
    assert isinstance(json_str, str)
    assert "Latency" in json_str
    assert "pass_rate" in json_str
