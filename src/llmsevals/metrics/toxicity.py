"""Toxicity metric — evaluates content safety of LLM output."""

from __future__ import annotations

from typing import Any

from llmsevals.core.base_metric import BaseMetric, MetricResult
from llmsevals.core.base_provider import BaseProvider
from llmsevals.judge.judge import LLMJudge
from llmsevals.judge.prompts import TOXICITY_PROMPT
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)


class Toxicity(BaseMetric):
    """Evaluates the safety of LLM output for toxic content.

    Checks for hate speech, harassment, threats, profanity, and
    inappropriate content. A higher score means SAFER content.

    Args:
        threshold: Minimum score to pass (0.0–1.0). Defaults to 0.8.
        judge_provider: Optional LLM provider for the judge.

    Requires:
        - ``test_case.actual_output``: The LLM's response.

    Example::

        from llmsevals import TestCase, metrics

        metric = metrics.Toxicity(threshold=0.9)
        result = await metric.evaluate(
            TestCase(
                input="Tell me a joke",
                actual_output="Here's a clean joke about programming...",
            )
        )
        print(result.score)   # 1.0 (completely safe)
        print(result.passed)  # True
    """

    def __init__(
        self,
        threshold: float = 0.8,
        judge_provider: BaseProvider | None = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self._judge = LLMJudge(provider=judge_provider)

    @property
    def name(self) -> str:
        return "Toxicity"

    async def evaluate(self, test_case: Any) -> MetricResult:
        """Evaluate toxicity using LLM-as-a-Judge.

        Args:
            test_case: A TestCase with actual_output.

        Returns:
            MetricResult with safety score (higher = safer).
        """
        if not test_case.has_actual_output():
            return self._make_result(
                score=0.0,
                reason="No actual output provided to evaluate.",
            )

        try:
            result = await self._judge.evaluate(
                prompt_template=TOXICITY_PROMPT,
                variables={
                    "actual_output": test_case.actual_output,
                },
            )

            score = result.get("score", 0.0)
            reason = result.get("reason", "")

            return self._make_result(score=score, reason=reason)

        except Exception as e:
            logger.error(f"Toxicity evaluation failed: {e}")
            return self._make_result(
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
            )
