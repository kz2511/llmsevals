"""Faithfulness metric — measures if the answer is faithful to the provided context."""

from __future__ import annotations

from typing import Any

from llmsevals.core.base_metric import BaseMetric, MetricResult
from llmsevals.core.base_provider import BaseProvider
from llmsevals.judge.judge import LLMJudge
from llmsevals.judge.prompts import FAITHFULNESS_PROMPT
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)


class Faithfulness(BaseMetric):
    """Measures whether the LLM's answer is faithful to the provided context.

    This is the most critical metric for RAG (Retrieval-Augmented Generation)
    systems. A faithful answer only contains information that can be derived
    from the provided context — it does not hallucinate or add unsupported facts.

    Args:
        threshold: Minimum score to pass (0.0–1.0). Defaults to 0.7.
        judge_provider: Optional LLM provider for the judge.

    Requires:
        - ``test_case.input``: The question/prompt.
        - ``test_case.actual_output``: The LLM's response.
        - ``test_case.context``: List of context documents (retrieved chunks).

    Example::

        from llmsevals import TestCase, metrics

        metric = metrics.Faithfulness(threshold=0.8)
        result = await metric.evaluate(
            TestCase(
                input="When was Python created?",
                actual_output="Python was created in 1991 by Guido van Rossum.",
                context=["Python was first released in 1991. Its creator is Guido van Rossum."],
            )
        )
        print(result.score)   # 1.0
        print(result.passed)  # True
    """

    def __init__(
        self,
        threshold: float = 0.7,
        judge_provider: BaseProvider | None = None,
        max_context_tokens: int = 4000,
    ) -> None:
        super().__init__(threshold=threshold)
        self._judge = LLMJudge(provider=judge_provider)
        self.max_context_tokens = max_context_tokens

    @property
    def name(self) -> str:
        return "Faithfulness"

    @property
    def required_fields(self) -> list[str]:
        return ["actual_output", "context"]

    async def evaluate(self, test_case: Any) -> MetricResult:
        """Evaluate faithfulness using LLM-as-a-Judge.

        Args:
            test_case: A TestCase with input, actual_output, and context.

        Returns:
            MetricResult with faithfulness score and explanation.
        """
        if not test_case.has_actual_output():
            return self._make_result(
                score=0.0,
                reason="No actual output provided to evaluate.",
            )

        if not test_case.has_context():
            return self._make_result(
                score=0.0,
                reason="No context provided. Faithfulness requires context for evaluation.",
                metadata={"warning": "missing_context"},
            )

        # Join context documents into a single string
        context_str = "\n\n---\n\n".join(test_case.context)

        # Truncate to token budget to prevent overflow
        from llmsevals.utils.sanitizer import truncate_to_tokens

        context_str = truncate_to_tokens(context_str, self.max_context_tokens)

        try:
            result = await self._judge.evaluate(
                prompt_template=FAITHFULNESS_PROMPT,
                variables={
                    "input": test_case.input,
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
            logger.error(f"Faithfulness evaluation failed: {e}")
            return self._make_result(
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
            )
