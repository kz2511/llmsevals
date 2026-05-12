"""Answer Relevancy metric — measures how relevant an answer is to the question."""

from __future__ import annotations

from typing import Any

from llmsevals.core.base_metric import BaseMetric, MetricResult
from llmsevals.core.base_provider import BaseProvider
from llmsevals.judge.judge import LLMJudge
from llmsevals.judge.prompts import ANSWER_RELEVANCY_PROMPT
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)


class AnswerRelevancy(BaseMetric):
    """Measures how relevant the LLM's answer is to the input question.

    Uses LLM-as-a-Judge to evaluate whether the response directly
    addresses the question asked.

    Args:
        threshold: Minimum score to pass (0.0–1.0). Defaults to 0.5.
        judge_provider: Optional LLM provider for the judge. If None,
            uses the default (gpt-4o-mini).

    Requires:
        - ``test_case.input``: The question/prompt.
        - ``test_case.actual_output``: The LLM's response.

    Example::

        from llmsevals import TestCase, metrics

        metric = metrics.AnswerRelevancy(threshold=0.7)
        result = await metric.evaluate(
            TestCase(
                input="What is Python?",
                actual_output="Python is a programming language.",
            )
        )
        print(result.score)   # 0.95
        print(result.passed)  # True
    """

    def __init__(
        self,
        threshold: float = 0.5,
        judge_provider: BaseProvider | None = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self._judge = LLMJudge(provider=judge_provider)

    @property
    def name(self) -> str:
        return "Answer Relevancy"

    async def evaluate(self, test_case: Any) -> MetricResult:
        """Evaluate answer relevancy using LLM-as-a-Judge.

        Args:
            test_case: A TestCase with input and actual_output.

        Returns:
            MetricResult with relevancy score and explanation.
        """
        if not test_case.has_actual_output():
            return self._make_result(
                score=0.0,
                reason="No actual output provided to evaluate.",
            )

        try:
            result = await self._judge.evaluate(
                prompt_template=ANSWER_RELEVANCY_PROMPT,
                variables={
                    "input": test_case.input,
                    "actual_output": test_case.actual_output,
                },
            )

            score = result.get("score", 0.0)
            reason = result.get("reason", "")

            return self._make_result(score=score, reason=reason)

        except Exception as e:
            logger.error(f"Answer Relevancy evaluation failed: {e}")
            return self._make_result(
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
            )
