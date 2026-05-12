"""Tests for JSON, CSV, and HTML file reporters."""

from __future__ import annotations

import json

import pytest

from llmsevals.core.base_metric import MetricResult
from llmsevals.core.eval_result import EvalResult, TestCaseResult
from llmsevals.reporters.csv_reporter import CSVReporter
from llmsevals.reporters.html_reporter import HTMLReporter
from llmsevals.reporters.json_reporter import JSONReporter


def _make_result(with_gen_error: bool = False) -> EvalResult:
    tcr = TestCaseResult(
        test_case_index=0,
        input="What is Python?",
        actual_output="A programming language." if not with_gen_error else None,
        metric_results=(
            []
            if with_gen_error
            else [MetricResult(name="Latency", score=0.9, passed=True, reason="Fast")]
        ),
        generation_error="API timeout" if with_gen_error else None,
    )
    return EvalResult(
        test_case_results=[tcr],
        metadata={"run_id": "test-001", "metrics": ["Latency"]},
    )


# =============================================================================
# JSONReporter
# =============================================================================


def test_json_reporter_creates_file(tmp_path):
    path = tmp_path / "out.json"
    JSONReporter().report(_make_result(), path)
    assert path.exists()


def test_json_reporter_valid_json(tmp_path):
    path = tmp_path / "out.json"
    JSONReporter().report(_make_result(), path)
    data = json.loads(path.read_text())
    assert "test_case_results" in data
    assert data["total_test_cases"] == 1


def test_json_reporter_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "out.json"
    JSONReporter().report(_make_result(), path)
    assert path.exists()


def test_json_reporter_includes_generation_error(tmp_path):
    path = tmp_path / "out.json"
    JSONReporter().report(_make_result(with_gen_error=True), path)
    data = json.loads(path.read_text())
    assert data["test_case_results"][0]["generation_error"] == "API timeout"


# =============================================================================
# CSVReporter
# =============================================================================


def test_csv_reporter_creates_file(tmp_path):
    path = tmp_path / "out.csv"
    CSVReporter().report(_make_result(), path)
    assert path.exists()


def test_csv_reporter_has_correct_headers(tmp_path):
    path = tmp_path / "out.csv"
    CSVReporter().report(_make_result(), path)
    lines = path.read_text().splitlines()
    header = lines[0]
    assert "index" in header
    assert "input" in header
    assert "passed" in header
    assert "Latency_score" in header
    assert "Latency_passed" in header


def test_csv_reporter_has_data_row(tmp_path):
    path = tmp_path / "out.csv"
    CSVReporter().report(_make_result(), path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2  # header + 1 data row


def test_csv_reporter_empty_result_writes_empty_file(tmp_path):
    path = tmp_path / "out.csv"
    CSVReporter().report(EvalResult(test_case_results=[]), path)
    assert path.exists()
    assert path.read_text() == ""


def test_csv_reporter_creates_parent_dirs(tmp_path):
    path = tmp_path / "sub" / "out.csv"
    CSVReporter().report(_make_result(), path)
    assert path.exists()


def test_csv_reporter_generation_error_in_row(tmp_path):
    path = tmp_path / "out.csv"
    CSVReporter().report(_make_result(with_gen_error=True), path)
    content = path.read_text()
    assert "API timeout" in content


# =============================================================================
# HTMLReporter
# =============================================================================


def test_html_reporter_creates_file(tmp_path):
    path = tmp_path / "out.html"
    HTMLReporter().report(_make_result(), path)
    assert path.exists()


def test_html_reporter_is_valid_html(tmp_path):
    path = tmp_path / "out.html"
    HTMLReporter().report(_make_result(), path)
    content = path.read_text()
    assert "<!DOCTYPE html>" in content
    assert "<html" in content
    assert "</html>" in content


def test_html_reporter_contains_metric_name(tmp_path):
    path = tmp_path / "out.html"
    HTMLReporter().report(_make_result(), path)
    content = path.read_text()
    assert "Latency" in content


def test_html_reporter_contains_run_id(tmp_path):
    path = tmp_path / "out.html"
    HTMLReporter().report(_make_result(), path)
    content = path.read_text()
    assert "test-001" in content


def test_html_reporter_escapes_special_chars(tmp_path):
    result = EvalResult(
        test_case_results=[
            TestCaseResult(
                test_case_index=0,
                input="<script>alert('xss')</script>",
                actual_output="Safe output",
                metric_results=[],
            )
        ]
    )
    path = tmp_path / "out.html"
    HTMLReporter().report(result, path)
    content = path.read_text()
    assert "<script>" not in content
    assert "&lt;script&gt;" in content


def test_html_reporter_shows_gen_error_amber(tmp_path):
    path = tmp_path / "out.html"
    HTMLReporter().report(_make_result(with_gen_error=True), path)
    content = path.read_text()
    assert "GEN ERROR" in content


def test_html_reporter_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "out.html"
    HTMLReporter().report(_make_result(), path)
    assert path.exists()


# =============================================================================
# reporters __init__ exports
# =============================================================================


def test_all_reporters_importable_from_package():
    from llmsevals.reporters import CSVReporter, HTMLReporter, JSONReporter
    assert CSVReporter is not None
    assert HTMLReporter is not None
    assert JSONReporter is not None
