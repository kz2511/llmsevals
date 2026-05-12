"""Tests for the ConsoleReporter."""

from __future__ import annotations

from llmsevals.core.base_metric import MetricResult
from llmsevals.core.eval_result import EvalResult, TestCaseResult
from llmsevals.reporters.console import ConsoleReporter


def test_console_reporter_runs_without_error():
    """ConsoleReporter.report() should run without raising."""
    result = EvalResult(
        test_case_results=[
            TestCaseResult(
                test_case_index=0,
                input="What is Python?",
                actual_output="Python is a programming language.",
                metric_results=[
                    MetricResult(
                        name="Latency",
                        score=0.85,
                        passed=True,
                        reason="Response time: 750ms",
                    ),
                ],
            ),
        ],
        metadata={"total_time_ms": 1234.5, "metrics": ["Latency"]},
    )

    reporter = ConsoleReporter()
    # Should not raise
    reporter.report(result)


def test_console_reporter_empty_results():
    """Should handle empty results gracefully."""
    result = EvalResult()
    reporter = ConsoleReporter()
    reporter.report(result)


def test_console_reporter_failed_results():
    """Should handle failed results correctly."""
    result = EvalResult(
        test_case_results=[
            TestCaseResult(
                test_case_index=0,
                input="Test question",
                actual_output="Bad answer",
                metric_results=[
                    MetricResult(
                        name="Faithfulness",
                        score=0.2,
                        passed=False,
                        reason="Answer not supported by context.",
                    ),
                ],
            ),
        ],
    )

    reporter = ConsoleReporter()
    reporter.report(result)
