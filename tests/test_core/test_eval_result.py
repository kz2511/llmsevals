"""Tests for the EvalResult data model."""

from __future__ import annotations

import json

from llmsevals.core.base_metric import MetricResult
from llmsevals.core.eval_result import EvalResult, TestCaseResult


def _make_tcr(index: int, passed: bool, score: float) -> TestCaseResult:
    """Helper to create a TestCaseResult."""
    return TestCaseResult(
        test_case_index=index,
        input=f"Question {index}",
        actual_output=f"Answer {index}",
        metric_results=[
            MetricResult(
                name="TestMetric",
                score=score,
                passed=passed,
                reason=f"Score: {score}",
            )
        ],
    )


def test_eval_result_empty():
    """Empty EvalResult should have zero counts."""
    er = EvalResult()
    assert er.total_test_cases == 0
    assert er.passed_count == 0
    assert er.failed_count == 0
    assert er.pass_rate == 0.0
    assert er.average_score == 0.0


def test_eval_result_all_pass():
    """Should count all passed correctly."""
    er = EvalResult(
        test_case_results=[
            _make_tcr(0, True, 0.9),
            _make_tcr(1, True, 0.8),
        ]
    )
    assert er.total_test_cases == 2
    assert er.passed_count == 2
    assert er.failed_count == 0
    assert er.pass_rate == 1.0


def test_eval_result_mixed():
    """Should count mixed pass/fail correctly."""
    er = EvalResult(
        test_case_results=[
            _make_tcr(0, True, 0.9),
            _make_tcr(1, False, 0.3),
            _make_tcr(2, True, 0.7),
        ]
    )
    assert er.total_test_cases == 3
    assert er.passed_count == 2
    assert er.failed_count == 1
    assert abs(er.pass_rate - 2 / 3) < 0.01


def test_eval_result_average_score():
    """Should calculate average score correctly."""
    er = EvalResult(
        test_case_results=[
            _make_tcr(0, True, 0.8),
            _make_tcr(1, True, 0.6),
        ]
    )
    assert abs(er.average_score - 0.7) < 0.01


def test_eval_result_metric_summary():
    """metric_summary() should aggregate per-metric stats."""
    er = EvalResult(
        test_case_results=[
            _make_tcr(0, True, 0.9),
            _make_tcr(1, False, 0.3),
        ]
    )
    summary = er.metric_summary()

    assert "TestMetric" in summary
    assert abs(summary["TestMetric"]["avg_score"] - 0.6) < 0.01
    assert summary["TestMetric"]["pass_rate"] == 0.5
    assert summary["TestMetric"]["min_score"] == 0.3
    assert summary["TestMetric"]["max_score"] == 0.9


def test_eval_result_to_dict():
    """to_dict() should return a serializable dictionary."""
    er = EvalResult(
        test_case_results=[_make_tcr(0, True, 0.8)],
        metadata={"model": "gpt-4o"},
    )
    d = er.to_dict()

    assert d["total_test_cases"] == 1
    assert d["passed"] == 1
    assert d["failed"] == 0
    assert "test_case_results" in d
    assert d["metadata"]["model"] == "gpt-4o"


def test_eval_result_to_json():
    """to_json() should return valid JSON."""
    er = EvalResult(
        test_case_results=[_make_tcr(0, True, 0.8)],
    )
    json_str = er.to_json()

    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["total_test_cases"] == 1


def test_testcase_result_get_metric():
    """get_metric() should find by name case-insensitively."""
    tcr = _make_tcr(0, True, 0.8)

    assert tcr.get_metric("TestMetric") is not None
    assert tcr.get_metric("testmetric") is not None
    assert tcr.get_metric("NonExistent") is None


def test_testcase_result_score_average():
    """TestCaseResult.score should average across metrics."""
    tcr = TestCaseResult(
        test_case_index=0,
        input="Q",
        actual_output="A",
        metric_results=[
            MetricResult(name="M1", score=0.8, passed=True, reason=""),
            MetricResult(name="M2", score=0.6, passed=True, reason=""),
        ],
    )
    assert abs(tcr.score - 0.7) < 0.01


def test_testcase_result_passed_all_metrics():
    """passed should be True only if ALL metrics passed."""
    tcr_pass = TestCaseResult(
        test_case_index=0,
        input="Q",
        metric_results=[
            MetricResult(name="M1", score=0.8, passed=True, reason=""),
            MetricResult(name="M2", score=0.7, passed=True, reason=""),
        ],
    )
    assert tcr_pass.passed is True

    tcr_fail = TestCaseResult(
        test_case_index=0,
        input="Q",
        metric_results=[
            MetricResult(name="M1", score=0.8, passed=True, reason=""),
            MetricResult(name="M2", score=0.3, passed=False, reason=""),
        ],
    )
    assert tcr_fail.passed is False
