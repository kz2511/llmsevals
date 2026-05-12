"""Integration tests — full pipeline with mock provider."""

from __future__ import annotations

import pytest

from llmsevals.core.base_provider import BaseProvider, GenerationResult
from llmsevals.core.dataset import Dataset
from llmsevals.core.eval_result import EvalResult
from llmsevals.core.test_case import TestCase
from llmsevals.metrics.latency import Latency
from llmsevals.metrics.cost import Cost
from llmsevals.runners.evaluator import Evaluator


class MockProvider(BaseProvider):
    """A mock provider that returns fixed responses for integration tests."""

    def __init__(self):
        super().__init__(model="mock-model")

    async def generate(self, prompt: str) -> GenerationResult:
        return GenerationResult(
            text=f"Mock response to: {prompt[:30]}",
            latency_ms=150.0,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            model="mock-model",
        )

    async def generate_with_messages(self, messages):
        return await self.generate(messages[-1]["content"])


# =============================================================================
# Full pipeline integration tests
# =============================================================================


def test_full_pipeline_latency_cost():
    """Full pipeline: Evaluator → Latency + Cost → ConsoleReporter → JSON."""
    evaluator = Evaluator(
        metrics=[
            Latency(threshold=0.5, max_latency_ms=5000),
            Cost(model="gpt-4o-mini", max_cost_usd=0.10),
        ],
        verbose=False,
    )

    test_cases = [
        TestCase(
            input="What is AI?",
            actual_output="AI is artificial intelligence.",
            latency_ms=800.0,
            token_count={"prompt_tokens": 10, "completion_tokens": 30},
        ),
        TestCase(
            input="What is Python?",
            actual_output="Python is a programming language.",
            latency_ms=1200.0,
            token_count={"prompt_tokens": 12, "completion_tokens": 25},
        ),
    ]

    result = evaluator.run(test_cases=test_cases)

    # Validate structure
    assert isinstance(result, EvalResult)
    assert result.total_test_cases == 2
    assert len(result.test_case_results) == 2

    # Each test case should have 2 metric results
    for tcr in result.test_case_results:
        assert len(tcr.metric_results) == 2
        metric_names = {mr.name for mr in tcr.metric_results}
        assert "Latency" in metric_names
        assert "Cost" in metric_names

    # JSON serialization should work
    json_str = result.to_json()
    assert "Latency" in json_str
    assert "Cost" in json_str
    assert "pass_rate" in json_str

    # Dict serialization should work
    d = result.to_dict()
    assert d["total_test_cases"] == 2
    assert "metric_summary" in d

    # Summary should not raise
    result.summary()


def test_full_pipeline_with_auto_generation():
    """Full pipeline with provider auto-generating outputs."""
    evaluator = Evaluator(
        provider=MockProvider(),
        metrics=[Latency(max_latency_ms=5000)],
        verbose=False,
    )

    result = evaluator.run(test_cases=[
        TestCase(input="Generate me something"),
    ])

    assert result.total_test_cases == 1
    tcr = result.test_case_results[0]
    assert tcr.actual_output is not None
    assert "Mock response" in tcr.actual_output


def test_full_pipeline_from_dataset():
    """Full pipeline loading from a Dataset object."""
    ds = Dataset([
        TestCase(input="Q1", actual_output="A1", latency_ms=500.0),
        TestCase(input="Q2", actual_output="A2", latency_ms=1500.0),
    ])

    evaluator = Evaluator(
        metrics=[Latency(threshold=0.5, max_latency_ms=5000)],
        verbose=False,
    )

    result = evaluator.run(dataset=ds)
    assert result.total_test_cases == 2


def test_parallel_execution_produces_same_results():
    """Parallel metric execution should produce same results as sequential."""
    test_cases = [
        TestCase(input="Q1", actual_output="A1", latency_ms=500.0,
                 token_count={"prompt_tokens": 10, "completion_tokens": 20}),
    ]
    metrics = [
        Latency(max_latency_ms=5000),
        Cost(model="gpt-4o-mini", max_cost_usd=0.10),
    ]

    # Parallel (default)
    evaluator_parallel = Evaluator(metrics=metrics, verbose=False)
    result_parallel = evaluator_parallel.run(test_cases=test_cases)

    # Sequential (max_concurrency=1)
    evaluator_seq = Evaluator(
        metrics=metrics, verbose=False, max_concurrency=1
    )
    result_seq = evaluator_seq.run(test_cases=test_cases)

    # Results should be identical
    assert result_parallel.total_test_cases == result_seq.total_test_cases
    for tcr_p, tcr_s in zip(
        result_parallel.test_case_results, result_seq.test_case_results
    ):
        for mr_p, mr_s in zip(tcr_p.metric_results, tcr_s.metric_results):
            assert mr_p.name == mr_s.name
            assert mr_p.score == mr_s.score
            assert mr_p.passed == mr_s.passed


def test_dry_run_cost_estimation():
    """Dry run should return cost estimation without running metrics."""
    evaluator = Evaluator(
        metrics=[Latency(max_latency_ms=5000)],
        verbose=False,
    )

    result = evaluator.run(
        test_cases=[
            TestCase(input="What is AI?", actual_output="AI is..."),
        ],
        dry_run=True,
    )

    assert result.metadata.get("dry_run") is True
    assert result.metadata.get("test_case_count") == 1
    assert len(result.test_case_results) == 0  # No actual evaluations
