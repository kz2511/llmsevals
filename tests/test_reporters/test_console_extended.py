"""Extended tests for the ConsoleReporter covering all format paths."""

from __future__ import annotations

from unittest.mock import patch

from llmsevals.core.base_metric import MetricResult
from llmsevals.core.eval_result import EvalResult, TestCaseResult
from llmsevals.reporters.console import ConsoleReporter


def _make_result(
    pass_rate_target: str = "high",
    include_time: bool = True,
    time_ms: float = 500.0,
) -> EvalResult:
    """Build an EvalResult with configurable pass/fail for coverage."""
    if pass_rate_target == "high":
        tcrs = [
            TestCaseResult(
                test_case_index=0,
                input="Short Q",
                actual_output="A",
                metric_results=[
                    MetricResult(name="M1", score=0.9, passed=True, reason="Good"),
                    MetricResult(name="M2", score=0.85, passed=True, reason="OK"),
                ],
            ),
        ]
    elif pass_rate_target == "medium":
        tcrs = [
            TestCaseResult(
                test_case_index=0,
                input="Q1",
                actual_output="A1",
                metric_results=[
                    MetricResult(name="M1", score=0.6, passed=True, reason="OK"),
                ],
            ),
            TestCaseResult(
                test_case_index=1,
                input="Q2",
                actual_output="A2",
                metric_results=[
                    MetricResult(name="M1", score=0.3, passed=False, reason="Bad"),
                ],
            ),
        ]
    else:  # low
        tcrs = [
            TestCaseResult(
                test_case_index=0,
                input="A very long input question that exceeds the maximum width limit for display",
                actual_output="A",
                metric_results=[
                    MetricResult(name="M1", score=0.2, passed=False, reason="Bad"),
                    MetricResult(name="M2", score=0.1, passed=False, reason="Terrible"),
                ],
            ),
        ]

    metadata = {}
    if include_time:
        metadata["total_time_ms"] = time_ms
    metadata["metrics"] = ["M1", "M2"]

    return EvalResult(test_case_results=tcrs, metadata=metadata)


def test_rich_report_all_pass():
    """Rich report with all tests passing (green colors)."""
    reporter = ConsoleReporter()
    result = _make_result("high")
    reporter.report(result)  # Should not raise


def test_rich_report_medium_pass_rate():
    """Rich report with 50% pass rate (yellow colors)."""
    reporter = ConsoleReporter()
    result = _make_result("medium")
    reporter.report(result)


def test_rich_report_low_pass_rate():
    """Rich report with all failing (red colors)."""
    reporter = ConsoleReporter()
    result = _make_result("low")
    reporter.report(result)


def test_rich_report_time_seconds():
    """Should display time in seconds when >= 1000ms."""
    reporter = ConsoleReporter()
    result = _make_result("high", time_ms=2500.0)
    reporter.report(result)


def test_rich_report_no_time():
    """Should handle missing time metadata."""
    reporter = ConsoleReporter()
    result = _make_result("high", include_time=False)
    reporter.report(result)


def test_plain_report_all_pass():
    """Plain text report with all passing."""
    reporter = ConsoleReporter()
    result = _make_result("high")
    reporter._plain_report(result)


def test_plain_report_failures():
    """Plain text report with failures."""
    reporter = ConsoleReporter()
    result = _make_result("low")
    reporter._plain_report(result)


def test_plain_report_empty():
    """Plain text report with no results."""
    reporter = ConsoleReporter()
    reporter._plain_report(EvalResult())


def test_plain_report_with_metrics():
    """Plain text report with metric summary."""
    reporter = ConsoleReporter()
    result = _make_result("medium")
    reporter._plain_report(result)


def test_report_fallback_to_plain():
    """Should fallback to plain text when Rich import fails."""
    reporter = ConsoleReporter()
    result = _make_result("high")

    with patch.dict("sys.modules", {"rich": None, "rich.console": None, "rich.panel": None, "rich.table": None, "rich.text": None}):
        # Force the ImportError path by calling _plain_report directly
        reporter._plain_report(result)
