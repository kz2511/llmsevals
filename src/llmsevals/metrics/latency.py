"""Latency metric — measures response time."""

from __future__ import annotations

from typing import Any

from llmsevals.core.base_metric import BaseMetric, MetricResult


class Latency(BaseMetric):
    """Measures the response latency of an LLM call.

    This is a non-LLM metric — it uses the latency data already recorded
    in the test case (populated by the provider during generation).

    The score is inversely proportional to latency: lower latency = higher score.

    Args:
        threshold: Minimum score to pass (0.0–1.0). Defaults to 0.5.
        max_latency_ms: The latency (in ms) that maps to a score of 0.0.
            Defaults to 10000 (10 seconds). Responses faster than this
            get linearly higher scores.

    Requires:
        - ``test_case.latency_ms``: Response latency in milliseconds.
            Auto-populated when using an Evaluator with a provider.

    Example::

        from llmsevals import TestCase, metrics

        metric = metrics.Latency(max_latency_ms=5000)
        result = await metric.evaluate(
            TestCase(
                input="test",
                actual_output="response",
                latency_ms=1500,
            )
        )
        print(result.score)   # 0.7 (1500ms out of 5000ms max)
        print(result.passed)  # True
    """

    def __init__(
        self,
        threshold: float = 0.5,
        max_latency_ms: float = 10000.0,
    ) -> None:
        super().__init__(threshold=threshold)
        self.max_latency_ms = max_latency_ms

    @property
    def name(self) -> str:
        return "Latency"

    @property
    def required_fields(self) -> list[str]:
        return ["latency_ms"]

    async def evaluate(self, test_case: Any) -> MetricResult:
        """Evaluate response latency.

        Args:
            test_case: A TestCase with latency_ms populated.

        Returns:
            MetricResult with latency score (higher = faster).
        """
        latency_ms = test_case.latency_ms

        if latency_ms is None:
            return self._make_result(
                score=0.0,
                reason=(
                    "No latency data available. "
                    "Use an Evaluator with a provider to auto-populate."
                ),
                metadata={"warning": "missing_latency"},
            )

        # Score: linearly inversely proportional to latency
        # 0ms -> 1.0, max_latency_ms -> 0.0
        score = max(0.0, 1.0 - (latency_ms / self.max_latency_ms))
        score = min(1.0, score)  # Clamp to [0, 1]

        # Human-readable latency
        if latency_ms < 1000:
            latency_str = f"{latency_ms:.0f}ms"
        else:
            latency_str = f"{latency_ms / 1000:.2f}s"

        return self._make_result(
            score=round(score, 4),
            reason=f"Response time: {latency_str} (max allowed: {self.max_latency_ms:.0f}ms)",
            metadata={
                "latency_ms": latency_ms,
                "max_latency_ms": self.max_latency_ms,
            },
        )
