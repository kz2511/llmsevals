"""Cost metric — estimates the cost of an LLM call."""

from __future__ import annotations

from typing import Any

from llmsevals.core.base_metric import BaseMetric, MetricResult
from llmsevals.utils.pricing import estimate_cost, get_model_pricing
from llmsevals.utils.tokenizer import count_tokens


class Cost(BaseMetric):
    """Estimates the cost of an LLM API call based on token usage.

    Uses built-in pricing data for 25+ popular models. The score is
    inversely proportional to cost: cheaper calls get higher scores.

    Args:
        threshold: Minimum score to pass (0.0–1.0). Defaults to 0.5.
        max_cost_usd: The cost (in USD) that maps to a score of 0.0.
            Defaults to 0.10 ($0.10 per call). Calls cheaper than this
            get linearly higher scores.
        model: Model name for pricing lookup. If None, uses metadata
            from the test case.

    Requires:
        - ``test_case.actual_output``: For token counting (if token_count not available).
        - ``test_case.token_count``: Dict with prompt_tokens / completion_tokens.
        - ``test_case.metadata["model"]``: Model name for pricing lookup.

    Example::

        from llmsevals import TestCase, metrics

        metric = metrics.Cost(max_cost_usd=0.05, model="gpt-4o")
        result = await metric.evaluate(
            TestCase(
                input="What is AI?",
                actual_output="AI is artificial intelligence...",
                token_count={"prompt_tokens": 10, "completion_tokens": 50},
            )
        )
        print(result.score)     # 0.97
        print(result.metadata)  # {"cost_usd": 0.000525, ...}
    """

    def __init__(
        self,
        threshold: float = 0.5,
        max_cost_usd: float = 0.10,
        model: str | None = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self.max_cost_usd = max_cost_usd
        self.model = model

    @property
    def name(self) -> str:
        return "Cost"

    async def evaluate(self, test_case: Any) -> MetricResult:
        """Evaluate the cost of an LLM call.

        Args:
            test_case: A TestCase with token usage or text for counting.

        Returns:
            MetricResult with cost score and USD estimate.
        """
        # Determine model
        model = self.model or test_case.metadata.get("model", "")
        if not model:
            return self._make_result(
                score=0.0,
                reason="No model specified. Set model in Cost() or test_case.metadata['model'].",
                metadata={"warning": "missing_model"},
            )

        # Check if pricing is available
        pricing = get_model_pricing(model)
        if pricing is None:
            return self._make_result(
                score=0.5,
                reason=f"No pricing data available for model '{model}'.",
                metadata={"warning": "unknown_pricing", "model": model},
            )

        # Get token counts
        if test_case.token_count:
            prompt_tokens = test_case.token_count.get("prompt_tokens", 0)
            completion_tokens = test_case.token_count.get("completion_tokens", 0)
        else:
            # Estimate from text
            prompt_tokens = count_tokens(test_case.input, model)
            completion_tokens = count_tokens(
                test_case.actual_output or "", model
            )

        # Calculate cost
        cost_usd = estimate_cost(model, prompt_tokens, completion_tokens)
        if cost_usd is None:
            return self._make_result(
                score=0.5,
                reason=f"Could not estimate cost for model '{model}'.",
            )

        # Score: inversely proportional to cost
        score = max(0.0, 1.0 - (cost_usd / self.max_cost_usd))
        score = min(1.0, score)

        # Format cost string
        if cost_usd < 0.01:
            cost_str = f"${cost_usd:.6f}"
        else:
            cost_str = f"${cost_usd:.4f}"

        return self._make_result(
            score=round(score, 4),
            reason=f"Estimated cost: {cost_str} (max budget: ${self.max_cost_usd:.4f})",
            metadata={
                "cost_usd": cost_usd,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "max_cost_usd": self.max_cost_usd,
            },
        )
