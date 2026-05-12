"""
llmsevals — A lightweight, provider-agnostic Python package for evaluating LLMs.

Evaluate any LLM across quality, safety, cost, and latency dimensions
with a single unified API.

Quick Start::

    from llmsevals import Evaluator, TestCase, metrics

    test_case = TestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris",
    )

    evaluator = Evaluator(metrics=[metrics.AnswerRelevancy(), metrics.Latency()])
    results = evaluator.run(test_cases=[test_case])
    results.summary()
"""

# Submodule imports for convenience
from llmsevals import metrics, providers, reporters
from llmsevals._version import __version__
from llmsevals.core.base_metric import BaseMetric, MetricResult
from llmsevals.core.base_provider import BaseProvider
from llmsevals.core.dataset import Dataset
from llmsevals.core.eval_result import EvalResult, TestCaseResult
from llmsevals.core.test_case import TestCase
from llmsevals.runners.evaluator import Evaluator

__all__ = [
    "__version__",
    # Core
    "TestCase",
    "EvalResult",
    "TestCaseResult",
    "BaseMetric",
    "MetricResult",
    "BaseProvider",
    "Dataset",
    # Runners
    "Evaluator",
    # Submodules
    "metrics",
    "providers",
    "reporters",
]
