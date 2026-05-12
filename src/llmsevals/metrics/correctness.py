"""Answer Correctness metric — compares output against a ground-truth expected answer."""

from __future__ import annotations

from typing import Any

from llmsevals.core.base_metric import BaseMetric, MetricResult
from llmsevals.core.base_provider import BaseProvider
from llmsevals.judge.judge import LLMJudge
from llmsevals.judge.prompts import ANSWER_CORRECTNESS_PROMPT
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)


class AnswerCorrectness(BaseMetric):
    """Measures factual correctness by comparing output to a ground-truth answer.

    Uses LLM-as-a-Judge to evaluate whether the response captures the same
    key facts as the expected (reference) answer.

    Args:
        threshold: Minimum score to pass (0.0–1.0). Defaults to 0.7.
        judge_provider: Optional LLM provider for the judge. If None,
            uses the default (gpt-4o-mini).

    Requires:
        - ``test_case.input``: The question/prompt.
        - ``test_case.actual_output``: The LLM's response.
        - ``test_case.expected_output``: The ground-truth reference answer.

    Example::

        from llmsevals import TestCase, metrics

        metric = metrics.AnswerCorrectness(threshold=0.7)
        result = await metric.evaluate(
            TestCase(
                input="When was Python created?",
                actual_output="Python was created in 1991.",
                expected_output="Python was first released in 1991 by Guido van Rossum.",
            )
        )
        print(result.score)   # 0.85
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
        return "Answer Correctness"

    @property
    def required_fields(self) -> list[str]:
        return ["actual_output", "expected_output"]

    async def evaluate(self, test_case: Any) -> MetricResult:
        """Evaluate answer correctness against ground-truth expected output.

        Args:
            test_case: A TestCase with input, actual_output, and expected_output.

        Returns:
            MetricResult with correctness score and explanation.
        """
        if not test_case.has_actual_output():
            return self._make_result(
                score=0.0,
                reason="No actual output provided to evaluate.",
            )

        if not test_case.has_expected_output():
            return self._make_result(
                score=0.0,
                reason=(
                    "No expected_output provided. "
                    "AnswerCorrectness requires a reference answer."
                ),
                metadata={"warning": "missing_expected_output"},
            )

        try:
            result = await self._judge.evaluate(
                prompt_template=ANSWER_CORRECTNESS_PROMPT,
                variables={
                    "input": test_case.input,
                    "actual_output": test_case.actual_output,
                    "expected_output": test_case.expected_output,
                },
            )

            score = result.get("score", 0.0)
            reason = result.get("reason", "")

            return self._make_result(score=score, reason=reason)

        except Exception as e:
            logger.error(f"Answer Correctness evaluation failed: {e}")
            return self._make_result(
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
            )
