"""EvalResult — Container for evaluation results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llmsevals.core.base_metric import MetricResult


class TestCaseResult(BaseModel):
    """Results for a single test case across all metrics.

    Attributes:
        test_case_index: Index of the test case in the dataset.
        input: The original input/prompt.
        actual_output: The LLM's response.
        metric_results: List of MetricResult for each metric evaluated.
        passed: True if ALL metrics passed for this test case.
    """

    test_case_index: int
    input: str
    actual_output: str | None = None
    metric_results: list[MetricResult] = Field(default_factory=list)
    generation_error: str | None = None

    @property
    def evaluated(self) -> bool:
        """True only if at least one metric was run and generation did not fail."""
        return len(self.metric_results) > 0 and self.generation_error is None

    @property
    def passed(self) -> bool:
        """True if all metrics passed for this test case."""
        if not self.metric_results:
            return False
        return all(r.passed for r in self.metric_results)

    @property
    def score(self) -> float:
        """Average score across all metrics."""
        if not self.metric_results:
            return 0.0
        return sum(r.score for r in self.metric_results) / len(self.metric_results)

    def get_metric(self, name: str) -> MetricResult | None:
        """Get a specific metric result by name."""
        for r in self.metric_results:
            if r.name.lower() == name.lower():
                return r
        return None


class EvalResult(BaseModel):
    """Aggregated evaluation results across all test cases and metrics.

    Attributes:
        test_case_results: Per-test-case results.
        metadata: Run-level metadata (model name, timestamp, etc.).
    """

    test_case_results: list[TestCaseResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_test_cases(self) -> int:
        """Total number of test cases evaluated."""
        return len(self.test_case_results)

    @property
    def passed_count(self) -> int:
        """Number of test cases where all metrics passed."""
        return sum(1 for r in self.test_case_results if r.passed)

    @property
    def failed_count(self) -> int:
        """Number of test cases where at least one metric failed."""
        return self.total_test_cases - self.passed_count

    @property
    def generation_error_count(self) -> int:
        """Number of test cases where generation failed before any metric ran."""
        return sum(1 for r in self.test_case_results if r.generation_error is not None)

    @property
    def pass_rate(self) -> float:
        """Percentage of test cases that passed (0.0–1.0)."""
        if self.total_test_cases == 0:
            return 0.0
        return self.passed_count / self.total_test_cases

    @property
    def average_score(self) -> float:
        """Average score across all test cases."""
        if not self.test_case_results:
            return 0.0
        return sum(r.score for r in self.test_case_results) / len(self.test_case_results)

    def metric_summary(self) -> dict[str, dict[str, float]]:
        """Get per-metric aggregate statistics.

        Returns:
            Dict mapping metric name to {avg_score, pass_rate, min, max}.
        """
        metrics_data: dict[str, list[MetricResult]] = {}
        for tcr in self.test_case_results:
            for mr in tcr.metric_results:
                metrics_data.setdefault(mr.name, []).append(mr)

        summary: dict[str, dict[str, float]] = {}
        for name, results in metrics_data.items():
            scores = [r.score for r in results]
            passed = sum(1 for r in results if r.passed)
            summary[name] = {
                "avg_score": sum(scores) / len(scores),
                "pass_rate": passed / len(results),
                "min_score": min(scores),
                "max_score": max(scores),
                "count": float(len(results)),
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result to a dictionary."""
        return {
            "total_test_cases": self.total_test_cases,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "generation_errors": self.generation_error_count,
            "pass_rate": round(self.pass_rate, 4),
            "average_score": round(self.average_score, 4),
            "metric_summary": self.metric_summary(),
            "test_case_results": [
                {
                    "index": tcr.test_case_index,
                    "input": tcr.input,
                    "actual_output": tcr.actual_output,
                    "generation_error": tcr.generation_error,
                    "passed": tcr.passed,
                    "score": round(tcr.score, 4),
                    "metrics": [r.model_dump() for r in tcr.metric_results],
                }
                for tcr in self.test_case_results
            ],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize results to a JSON string. raw_response fields are always excluded."""
        import json

        def _safe_default(obj: object) -> object:
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            return f"<non-serializable: {type(obj).__name__}>"

        return json.dumps(self.to_dict(), indent=2, default=_safe_default)

    def summary(self) -> None:
        """Print a rich summary to the console."""
        from llmsevals.reporters.console import ConsoleReporter

        reporter = ConsoleReporter()
        reporter.report(self)
