"""Tests for BaseMetric.required_fields and TestCase.validate_for()."""

from __future__ import annotations

import pytest

from llmsevals.core.test_case import TestCase
from llmsevals.metrics.answer_relevancy import AnswerRelevancy
from llmsevals.metrics.correctness import AnswerCorrectness
from llmsevals.metrics.faithfulness import Faithfulness
from llmsevals.metrics.hallucination import Hallucination
from llmsevals.metrics.latency import Latency
from llmsevals.metrics.toxicity import Toxicity
from llmsevals.metrics.bias import Bias
from llmsevals.metrics.coherence import Coherence
from llmsevals.metrics.cost import Cost


# =============================================================================
# required_fields per metric
# =============================================================================


def test_answer_relevancy_required_fields():
    assert "actual_output" in AnswerRelevancy().required_fields


def test_faithfulness_required_fields():
    fields = Faithfulness().required_fields
    assert "actual_output" in fields
    assert "context" in fields


def test_hallucination_required_fields():
    fields = Hallucination().required_fields
    assert "actual_output" in fields
    assert "context" in fields


def test_answer_correctness_required_fields():
    fields = AnswerCorrectness().required_fields
    assert "actual_output" in fields
    assert "expected_output" in fields


def test_latency_required_fields():
    assert "latency_ms" in Latency().required_fields


def test_toxicity_required_fields():
    assert "actual_output" in Toxicity().required_fields


def test_bias_required_fields():
    assert "actual_output" in Bias().required_fields


def test_coherence_required_fields():
    assert "actual_output" in Coherence().required_fields


def test_cost_required_fields():
    assert "actual_output" in Cost(model="gpt-4o").required_fields


# =============================================================================
# validate_for()
# =============================================================================


def test_validate_for_all_present_faithfulness():
    tc = TestCase(input="Q", actual_output="A", context=["chunk"])
    missing = tc.validate_for(Faithfulness())
    assert missing == []


def test_validate_for_missing_context_faithfulness():
    tc = TestCase(input="Q", actual_output="A")
    missing = tc.validate_for(Faithfulness())
    assert "context" in missing


def test_validate_for_missing_actual_output():
    tc = TestCase(input="Q")
    missing = tc.validate_for(AnswerRelevancy())
    assert "actual_output" in missing


def test_validate_for_missing_expected_output():
    tc = TestCase(input="Q", actual_output="A")
    missing = tc.validate_for(AnswerCorrectness())
    assert "expected_output" in missing


def test_validate_for_missing_latency():
    tc = TestCase(input="Q", actual_output="A")
    missing = tc.validate_for(Latency())
    assert "latency_ms" in missing


def test_validate_for_latency_present():
    tc = TestCase(input="Q", actual_output="A", latency_ms=500.0)
    missing = tc.validate_for(Latency())
    assert missing == []


def test_validate_for_empty_actual_output_counts_as_missing():
    tc = TestCase(input="Q", actual_output="")
    missing = tc.validate_for(AnswerRelevancy())
    assert "actual_output" in missing


def test_validate_for_empty_context_counts_as_missing():
    tc = TestCase(input="Q", actual_output="A", context=[])
    missing = tc.validate_for(Faithfulness())
    assert "context" in missing


def test_validate_for_object_without_required_fields():
    """Metrics without required_fields should return empty missing list."""
    class _NoFields:
        pass
    tc = TestCase(input="Q")
    assert tc.validate_for(_NoFields()) == []
