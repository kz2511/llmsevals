"""Extended tests for the Evaluator to cover all remaining code paths."""

from __future__ import annotations

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmsevals.core.base_metric import BaseMetric, MetricResult
from llmsevals.core.base_provider import BaseProvider, GenerationResult
from llmsevals.core.dataset import Dataset
from llmsevals.core.test_case import TestCase
from llmsevals.metrics.latency import Latency
from llmsevals.runners.evaluator import Evaluator


# --- Mock provider for testing auto-generation ---

class MockProvider(BaseProvider):
    """A mock provider that returns fixed responses."""

    def __init__(self):
        super().__init__(model="mock-model")

    async def generate(self, prompt: str) -> GenerationResult:
        return GenerationResult(
            text=f"Mock response to: {prompt[:30]}",
            latency_ms=100.0,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            model="mock-model",
        )

    async def generate_with_messages(self, messages):
        return await self.generate(messages[-1]["content"])


class FailingProvider(BaseProvider):
    """A provider that always raises."""

    def __init__(self):
        super().__init__(model="failing-model")

    async def generate(self, prompt: str) -> GenerationResult:
        raise ConnectionError("Network unreachable")

    async def generate_with_messages(self, messages):
        raise ConnectionError("Network unreachable")


class FailingMetric(BaseMetric):
    """A metric that always raises."""

    @property
    def name(self) -> str:
        return "FailingMetric"

    async def evaluate(self, test_case) -> MetricResult:
        raise RuntimeError("Metric evaluation crashed")


# =============================================================================
# Auto-generation tests
# =============================================================================


def test_evaluator_auto_generate_with_provider():
    """Should auto-generate outputs when provider is set and actual_output is missing."""
    evaluator = Evaluator(
        provider=MockProvider(),
        metrics=[Latency(max_latency_ms=5000)],
        verbose=False,
    )

    test_cases = [
        TestCase(input="What is AI?"),  # No actual_output
        TestCase(input="What is ML?", actual_output="Machine Learning"),  # Has output
    ]

    result = evaluator.run(test_cases=test_cases)

    assert result.total_test_cases == 2
    # First test case should have auto-generated output
    tcr0 = result.test_case_results[0]
    assert tcr0.actual_output is not None
    assert "Mock response" in tcr0.actual_output

    # Second test case should keep original
    tcr1 = result.test_case_results[1]
    assert tcr1.actual_output == "Machine Learning"


def test_evaluator_auto_generate_failure():
    """Should handle generation failures gracefully."""
    evaluator = Evaluator(
        provider=FailingProvider(),
        metrics=[Latency(max_latency_ms=5000)],
        verbose=True,
    )

    test_cases = [TestCase(input="Test query")]
    result = evaluator.run(test_cases=test_cases)

    # Should still produce a result (even if output is None)
    assert result.total_test_cases == 1


# =============================================================================
# Metric error handling
# =============================================================================


def test_evaluator_metric_exception():
    """Should catch metric exceptions and record them as failed."""
    evaluator = Evaluator(
        metrics=[FailingMetric()],
        verbose=False,
    )

    result = evaluator.run(
        test_cases=[TestCase(input="Q1", actual_output="A1")]
    )

    assert result.total_test_cases == 1
    assert result.failed_count == 1
    tcr = result.test_case_results[0]
    assert tcr.metric_results[0].passed is False
    assert "Evaluation error" in tcr.metric_results[0].reason


# =============================================================================
# Dataset resolution tests
# =============================================================================


def test_evaluator_with_dataset_object():
    """Should accept a Dataset object."""
    ds = Dataset([
        TestCase(input="Q1", actual_output="A1", latency_ms=500.0),
        TestCase(input="Q2", actual_output="A2", latency_ms=1000.0),
    ])

    evaluator = Evaluator(
        metrics=[Latency(max_latency_ms=5000)],
        verbose=False,
    )
    result = evaluator.run(dataset=ds)
    assert result.total_test_cases == 2


@pytest.fixture
def tmp_dir():
    """Create temp dir for test files."""
    tmp = os.path.join(os.path.dirname(__file__), ".tmp_eval_data")
    os.makedirs(tmp, exist_ok=True)
    yield tmp
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def test_evaluator_with_json_dataset(tmp_dir):
    """Should load from a JSON file path."""
    path = os.path.join(tmp_dir, "test.json")
    with open(path, "w") as f:
        json.dump([
            {"input": "Q1", "actual_output": "A1", "latency_ms": 500},
        ], f)

    evaluator = Evaluator(
        metrics=[Latency(max_latency_ms=5000)],
        verbose=False,
    )
    result = evaluator.run(dataset=path)
    assert result.total_test_cases == 1


def test_evaluator_with_jsonl_dataset(tmp_dir):
    """Should load from a JSONL file path."""
    path = os.path.join(tmp_dir, "test.jsonl")
    with open(path, "w") as f:
        f.write('{"input": "Q1", "actual_output": "A1", "latency_ms": 500}\n')
        f.write('{"input": "Q2", "actual_output": "A2", "latency_ms": 800}\n')

    evaluator = Evaluator(
        metrics=[Latency(max_latency_ms=5000)],
        verbose=False,
    )
    result = evaluator.run(dataset=path)
    assert result.total_test_cases == 2


def test_evaluator_with_csv_dataset(tmp_dir):
    """Should load from a CSV file path."""
    path = os.path.join(tmp_dir, "test.csv")
    with open(path, "w") as f:
        f.write("input,actual_output\n")
        f.write("Q1,A1\n")

    evaluator = Evaluator(
        metrics=[Latency(max_latency_ms=5000)],
        verbose=False,
    )
    result = evaluator.run(dataset=path)
    assert result.total_test_cases == 1


def test_evaluator_invalid_dataset_type():
    """Should raise ValueError for invalid dataset type."""
    evaluator = Evaluator(
        metrics=[Latency()],
        verbose=False,
    )

    with pytest.raises(ValueError, match="Invalid dataset type"):
        evaluator.run(dataset=12345)


# =============================================================================
# Override metrics at run-time
# =============================================================================


def test_evaluator_run_override_metrics():
    """Should allow overriding metrics in run() call."""
    evaluator = Evaluator(
        metrics=[Latency(threshold=0.9, max_latency_ms=1000)],  # Strict
        verbose=False,
    )

    tc = TestCase(input="Q", actual_output="A", latency_ms=500.0)

    # Default metrics would fail (500ms out of 1000ms = 0.5 < 0.9)
    result1 = evaluator.run(test_cases=[tc])
    assert result1.test_case_results[0].passed is False

    # Override with lenient metric
    result2 = evaluator.run(
        test_cases=[tc],
        metrics=[Latency(threshold=0.3, max_latency_ms=5000)],
    )
    assert result2.test_case_results[0].passed is True


# =============================================================================
# Verbose logging paths
# =============================================================================


def test_evaluator_verbose_mode():
    """Verbose mode should log without errors."""
    evaluator = Evaluator(
        metrics=[Latency(max_latency_ms=5000)],
        verbose=True,
    )

    result = evaluator.run(
        test_cases=[TestCase(input="Q1", actual_output="A1", latency_ms=500.0)]
    )
    assert result.total_test_cases == 1


def test_evaluator_verbose_with_provider():
    """Verbose mode with provider should log generation."""
    evaluator = Evaluator(
        provider=MockProvider(),
        metrics=[Latency(max_latency_ms=5000)],
        verbose=True,
    )

    result = evaluator.run(
        test_cases=[TestCase(input="Generate for me please")]
    )
    assert result.total_test_cases == 1


# =============================================================================
# Repr and model shorthand
# =============================================================================


def test_evaluator_repr():
    """__repr__ should show metrics and provider."""
    evaluator = Evaluator(
        metrics=[Latency()],
        verbose=False,
    )
    r = repr(evaluator)
    assert "Evaluator" in r
    assert "Latency" in r


def test_evaluator_model_shorthand():
    """model= param should create an OpenAI provider."""
    evaluator = Evaluator(model="gpt-4o-mini", verbose=False)
    # Should have set the provider (openai is installed in our test env)
    assert evaluator.provider is not None
