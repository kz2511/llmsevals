"""Bias metric — evaluates fairness and bias in LLM output."""

from __future__ import annotations

from typing import Any

from llmsevals.core.base_metric import BaseMetric, MetricResult
from llmsevals.core.base_provider import BaseProvider
from llmsevals.judge.judge import LLMJudge
from llmsevals.judge.prompts import BIAS_PROMPT
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)


class Bias(BaseMetric):
    """Evaluates text for biases including gender, racial, and age bias.

    A higher score means LESS bias (more fair and inclusive). Checks for
    stereotypes, generalizations, and discriminatory language.

    Args:
        threshold: Minimum score to pass (0.0–1.0). Defaults to 0.7.
        judge_provider: Optional LLM provider for the judge.

    Requires:
        - ``test_case.actual_output``: The LLM's response.

    Example::

        from llmsevals import TestCase, metrics

        metric = metrics.Bias(threshold=0.8)
        result = await metric.evaluate(
            TestCase(
                input="Describe a software engineer",
                actual_output="A software engineer designs and builds software systems...",
            )
        )
        print(result.score)   # 0.95 (minimal bias)
        print(result.passed)  # True
    """

    def __init__(
        self,
        threshold: float = 0.7,
        judge_provider: BaseProvider | None = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self._judge = LLMJudge(provider=judge_provider)

    @property
    def name(self) -> str:
        return "Bias"

    async def evaluate(self, test_case: Any) -> MetricResult:
        """Evaluate bias using LLM-as-a-Judge.

        Args:
            test_case: A TestCase with actual_output.

        Returns:
            MetricResult with bias score (higher = less biased).
        """
        if not test_case.has_actual_output():
            return self._make_result(
                score=0.0,
                reason="No actual output provided to evaluate.",
            )

        try:
            result = await self._judge.evaluate(
                prompt_template=BIAS_PROMPT,
                variables={
                    "actual_output": test_case.actual_output,
                },
            )

            score = result.get("score", 0.0)
            reason = result.get("reason", "")

            return self._make_result(score=score, reason=reason)

        except Exception as e:
            logger.error(f"Bias evaluation failed: {e}")
            return self._make_result(
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
            )
