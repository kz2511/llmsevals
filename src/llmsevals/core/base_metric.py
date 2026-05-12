"""Base metric interface — all evaluation metrics inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Result from evaluating a single metric on a single test case.

    Attributes:
        name: The metric name (e.g. 'Faithfulness').
        score: Normalized score between 0.0 and 1.0.
        passed: Whether the score meets or exceeds the threshold.
        reason: Human-readable explanation of the score.
        metadata: Additional metric-specific data.
    """

    name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        status = "✅" if self.passed else "❌"
        return f"{status} {self.name}: {self.score:.2f} ({self.reason})"


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics.

    Every metric must implement:
        - ``name`` property: A human-readable metric name.
        - ``evaluate(test_case)``: The evaluation logic returning a MetricResult.

    Args:
        threshold: Minimum score (0.0–1.0) to consider the metric as "passed".
            Defaults to 0.5.

    Example::

        class MyCustomMetric(BaseMetric):
            @property
            def name(self) -> str:
                return "My Custom Metric"

            async def evaluate(self, test_case) -> MetricResult:
                score = my_scoring_logic(test_case)
                return MetricResult(
                    name=self.name,
                    score=score,
                    passed=score >= self.threshold,
                    reason="Scored based on custom logic.",
                )
    """

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.threshold = threshold

    @property
    def required_fields(self) -> list[str]:
        """TestCase fields that must be non-None/non-empty for this metric to evaluate.

        Override in subclasses to declare field requirements.
        Used by ``TestCase.validate_for()`` and ``Evaluator.dry_run()``.

        Returns:
            List of TestCase field names (e.g. ``["actual_output", "context"]``).
        """
        return ["actual_output"]

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this metric."""
        ...

    @abstractmethod
    async def evaluate(self, test_case: Any) -> MetricResult:
        """Evaluate a single test case and return a MetricResult.

        Args:
            test_case: A ``TestCase`` instance to evaluate.

        Returns:
            A ``MetricResult`` with score, pass/fail, and explanation.
        """
        ...

    def _make_result(
        self,
        score: float,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MetricResult:
        """Helper to construct a MetricResult with automatic pass/fail."""
        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            reason=reason,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"
