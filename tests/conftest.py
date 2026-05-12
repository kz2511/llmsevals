"""Shared test fixtures for llmsevals tests."""

from __future__ import annotations

import pytest

from llmsevals.core.test_case import TestCase


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "real_world: mark tests that require real API keys and external services"
    )
    config.addinivalue_line(
        "markers", "slow: mark tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "integration: mark tests as integration tests"
    )


@pytest.fixture
def simple_test_case() -> TestCase:
    """A simple Q&A test case."""
    return TestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris",
    )


@pytest.fixture
def rag_test_case() -> TestCase:
    """A RAG test case with context."""
    return TestCase(
        input="When was Python first released?",
        actual_output="Python was first released in 1991 by Guido van Rossum.",
        context=[
            "Python is a high-level programming language.",
            "Python was first released in 1991.",
            "Python was created by Guido van Rossum.",
        ],
    )


@pytest.fixture
def test_case_with_latency() -> TestCase:
    """A test case with latency data."""
    return TestCase(
        input="What is 2 + 2?",
        actual_output="The answer is 4.",
        latency_ms=1500.0,
    )


@pytest.fixture
def test_case_with_tokens() -> TestCase:
    """A test case with token count data."""
    return TestCase(
        input="Explain quantum computing briefly.",
        actual_output="Quantum computing uses quantum bits (qubits) to process information.",
        token_count={
            "prompt_tokens": 15,
            "completion_tokens": 45,
            "total_tokens": 60,
        },
        metadata={"model": "gpt-4o"},
    )


@pytest.fixture
def empty_test_case() -> TestCase:
    """A test case with no output."""
    return TestCase(input="What is AI?")
