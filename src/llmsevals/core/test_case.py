"""TestCase — The fundamental unit of evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single evaluation test case.

    This is the core data structure passed to every metric. Not all fields
    are required by every metric — for example, ``context`` is only needed
    for RAG metrics like Faithfulness.

    Attributes:
        input: The user query or prompt sent to the LLM.
        actual_output: The LLM's actual response. If not provided, the
            evaluator will generate it using the configured provider.
        expected_output: The ground-truth / reference answer (optional).
            Required for reference-based metrics like BLEU/ROUGE.
        context: Retrieved context documents (for RAG evaluation).
        conversation_history: Prior conversation turns (for multi-turn eval).
        metadata: Arbitrary key-value pairs for filtering/grouping results.
        latency_ms: Response latency in milliseconds (auto-populated by provider).
        token_count: Token usage details (auto-populated by provider).

    Example::

        test = TestCase(
            input="What is photosynthesis?",
            actual_output="Photosynthesis is the process by which plants...",
            context=["Photosynthesis is a biological process..."],
        )
    """

    input: str
    actual_output: str | None = None
    expected_output: str | None = None
    context: list[str] | None = None
    conversation_history: list[dict[str, str]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Auto-populated fields
    latency_ms: float | None = None
    token_count: dict[str, int] | None = None

    # Strict: unknown fields raise ValidationError, catching typos like contextt= instead of context=
    model_config = {"extra": "forbid"}

    def has_context(self) -> bool:
        """Check if context is provided (needed for RAG metrics)."""
        return self.context is not None and len(self.context) > 0

    def has_expected_output(self) -> bool:
        """Check if expected output is provided (needed for reference metrics)."""
        return self.expected_output is not None and len(self.expected_output) > 0

    def has_actual_output(self) -> bool:
        """Check if actual output is available."""
        return self.actual_output is not None and len(self.actual_output) > 0

    def validate_for(self, metric: object) -> list[str]:
        """Return list of fields missing for the given metric.

        Args:
            metric: A BaseMetric instance (uses its ``required_fields`` property).

        Returns:
            List of missing field names. Empty list means test case is valid for this metric.

        Example::

            missing = test_case.validate_for(Faithfulness())
            if missing:
                print(f"Missing fields: {missing}")  # ['context']
        """
        missing = []
        for field_name in getattr(metric, "required_fields", []):
            val = getattr(self, field_name, None)
            if val is None or (isinstance(val, (str, list)) and len(val) == 0):
                missing.append(field_name)
        return missing

    def __repr__(self) -> str:
        input_preview = self.input[:50] + "..." if len(self.input) > 50 else self.input
        return f"TestCase(input='{input_preview}')"
