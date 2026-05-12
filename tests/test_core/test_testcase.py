"""Tests for the TestCase data model."""

from __future__ import annotations

import pytest

from llmsevals.core.test_case import TestCase


def test_testcase_basic_creation():
    """Should create a TestCase with required fields."""
    tc = TestCase(input="What is AI?")
    assert tc.input == "What is AI?"
    assert tc.actual_output is None
    assert tc.expected_output is None
    assert tc.context is None
    assert tc.latency_ms is None
    assert tc.token_count is None


def test_testcase_full_creation():
    """Should create a TestCase with all fields."""
    tc = TestCase(
        input="What is Python?",
        actual_output="Python is a programming language.",
        expected_output="A programming language.",
        context=["Python was created in 1991."],
        latency_ms=500.0,
        token_count={"prompt_tokens": 10, "completion_tokens": 20},
        metadata={"model": "gpt-4o"},
    )
    assert tc.input == "What is Python?"
    assert tc.actual_output == "Python is a programming language."
    assert tc.expected_output == "A programming language."
    assert tc.context == ["Python was created in 1991."]
    assert tc.latency_ms == 500.0
    assert tc.token_count["prompt_tokens"] == 10
    assert tc.metadata["model"] == "gpt-4o"


def test_testcase_has_context():
    """has_context() should check for non-empty context."""
    tc_no_ctx = TestCase(input="Q")
    assert tc_no_ctx.has_context() is False

    tc_empty_ctx = TestCase(input="Q", context=[])
    assert tc_empty_ctx.has_context() is False

    tc_with_ctx = TestCase(input="Q", context=["Some context"])
    assert tc_with_ctx.has_context() is True


def test_testcase_has_expected_output():
    """has_expected_output() should check for non-empty expected output."""
    tc_no = TestCase(input="Q")
    assert tc_no.has_expected_output() is False

    tc_empty = TestCase(input="Q", expected_output="")
    assert tc_empty.has_expected_output() is False

    tc_yes = TestCase(input="Q", expected_output="Answer")
    assert tc_yes.has_expected_output() is True


def test_testcase_has_actual_output():
    """has_actual_output() should check for non-empty actual output."""
    tc_no = TestCase(input="Q")
    assert tc_no.has_actual_output() is False

    tc_empty = TestCase(input="Q", actual_output="")
    assert tc_empty.has_actual_output() is False

    tc_yes = TestCase(input="Q", actual_output="Response")
    assert tc_yes.has_actual_output() is True


def test_testcase_extra_fields_forbidden():
    """Unknown fields must raise ValidationError to catch typos like contextt=."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError) as exc_info:
        TestCase(input="Q", contextt=["chunk"])  # typo: 'contextt' not 'context'
    assert "contextt" in str(exc_info.value)


def test_testcase_correct_context_field_works():
    """The correct context= field must still work fine."""
    tc = TestCase(input="Q", context=["chunk"])
    assert tc.context == ["chunk"]


def test_testcase_repr():
    """__repr__ should show truncated input."""
    tc_short = TestCase(input="Short")
    assert "Short" in repr(tc_short)

    tc_long = TestCase(input="A" * 100)
    assert "..." in repr(tc_long)


def test_testcase_model_copy():
    """model_copy should create an updated copy."""
    tc = TestCase(input="Q1", actual_output="A1")
    tc2 = tc.model_copy(update={"actual_output": "A2"})

    assert tc.actual_output == "A1"
    assert tc2.actual_output == "A2"
    assert tc2.input == "Q1"
