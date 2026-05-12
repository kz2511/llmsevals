"""Hallucination metric — detects unsupported claims in LLM output."""

from __future__ import annotations

from typing import Any

from llmsevals.core.base_metric import BaseMetric, MetricResult
from llmsevals.core.base_provider import BaseProvider
from llmsevals.judge.judge import LLMJudge
from llmsevals.judge.prompts import HALLUCINATION_PROMPT
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)


class Hallucination(BaseMetric):
    """Detects hallucinations by comparing the answer against provided context.

    A hallucination is any claim, fact, or detail in the answer that is NOT
    supported by or contradicts the provided context. A higher score means
    LESS hallucination (more factually grounded).

    Args:
        threshold: Minimum score to pass (0.0–1.0). Defaults to 0.7.
        judge_provider: Optional LLM provider for the judge.

    Requires:
        - ``test_case.actual_output``: The LLM's response.
        - ``test_case.context``: List of context documents.

    Example::

        from llmsevals import TestCase, metrics

        metric = metrics.Hallucination(threshold=0.8)
        result = await metric.evaluate(
            TestCase(
                input="When was Python created?",
                actual_output="Python was created in 1995 by James Gosling.",
                context=["Python was first released in 1991 by Guido van Rossum."],
            )
        )
        print(result.score)   # 0.2 (significant hallucination)
        print(result.passed)  # False
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
        return "Hallucination"

    @property
    def required_fields(self) -> list[str]:
        return ["actual_output", "context"]

    async def evaluate(self, test_case: Any) -> MetricResult:
        """Evaluate hallucination using LLM-as-a-Judge.

        Args:
            test_case: A TestCase with actual_output and context.

        Returns:
            MetricResult with hallucination score (higher = less hallucination).
        """
        if not test_case.has_actual_output():
            return self._make_result(
                score=0.0,
                reason="No actual output provided to evaluate.",
            )

        if not test_case.has_context():
            return self._make_result(
                score=0.0,
                reason="No context provided. Hallucination detection requires context.",
                metadata={"warning": "missing_context"},
            )

        context_str = "\n\n---\n\n".join(test_case.context)

        try:
            result = await self._judge.evaluate(
                prompt_template=HALLUCINATION_PROMPT,
                variables={
                    "actual_output": test_case.actual_output,
                    "context": context_str,
                },
            )

            score = result.get("score", 0.0)
            reason = result.get("reason", "")

            return self._make_result(
                score=score,
                reason=reason,
                metadata={"context_chunks": len(test_case.context)},
            )

        except Exception as e:
            logger.error(f"Hallucination evaluation failed: {e}")
            return self._make_result(
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
            )
