"""Core abstractions for llmsevals."""

from llmsevals.core.base_metric import BaseMetric, MetricResult
from llmsevals.core.base_provider import BaseProvider
from llmsevals.core.dataset import Dataset
from llmsevals.core.eval_result import EvalResult, TestCaseResult
from llmsevals.core.test_case import TestCase

__all__ = [
    "BaseMetric",
    "MetricResult",
    "BaseProvider",
    "TestCase",
    "EvalResult",
    "TestCaseResult",
    "Dataset",
]
