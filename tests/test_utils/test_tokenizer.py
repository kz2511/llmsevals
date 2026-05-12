"""Tests for the tokenizer utility."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from llmsevals.utils.tokenizer import (
    clear_encoding_cache,
    count_message_tokens,
    count_tokens,
    _get_encoding,
)


def test_count_tokens_basic():
    """Should count tokens in a simple string."""
    count = count_tokens("Hello, world!")
    assert count > 0
    assert isinstance(count, int)


def test_count_tokens_empty():
    """Empty string should have 0 tokens."""
    count = count_tokens("")
    assert count == 0


def test_count_tokens_long_text():
    """Longer text should have more tokens."""
    short_count = count_tokens("Hello")
    long_count = count_tokens("Hello " * 100)
    assert long_count > short_count


def test_count_tokens_with_model():
    """Should accept a model parameter."""
    count = count_tokens("Hello, world!", model="gpt-4o")
    assert count > 0


def test_count_tokens_unknown_model():
    """Should fallback to default encoding for unknown models."""
    count = count_tokens("Hello, world!", model="unknown-model-xyz")
    assert count > 0


def test_count_message_tokens():
    """Should count tokens for chat messages."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AI?"},
    ]
    count = count_message_tokens(messages)
    assert count > 0

    # Should be more than just the content tokens (due to overhead)
    content_tokens = count_tokens("You are a helpful assistant.") + count_tokens("What is AI?")
    assert count > content_tokens


def test_count_message_tokens_empty():
    """Empty messages should have minimal tokens."""
    count = count_message_tokens([])
    assert count >= 0


def test_count_tokens_consistency():
    """Same text should give same token count."""
    text = "Quantum computing uses qubits."
    count1 = count_tokens(text)
    count2 = count_tokens(text)
    assert count1 == count2


# =============================================================================
# Missing coverage tests for 95%+ coverage
# =============================================================================


def test_clear_encoding_cache():
    """Should clear the tiktoken encoding cache."""
    # First, populate the cache
    _get_encoding("gpt-4o", use_model=True)
    assert _get_encoding.cache_info().currsize > 0

    # Clear the cache
    clear_encoding_cache()

    # Cache should be empty
    assert _get_encoding.cache_info().currsize == 0


def test_count_tokens_fallback_without_tiktoken():
    """Should use approximate token count when tiktoken is not available."""
    # Remove tiktoken from sys.modules temporarily
    original_tiktoken = sys.modules.get("tiktoken")

    with patch.dict(sys.modules, {"tiktoken": None}):
        with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: None if name == "tiktoken" else __builtins__.__import__(name, *args, **kwargs)):
            # This test is tricky because tiktoken is likely installed
            # We'll test by mocking the import to raise ImportError
            pass

    # Alternative approach: patch the try block to simulate ImportError
    with patch("llmsevals.utils.tokenizer._get_encoding") as mock_get_encoding:
        mock_get_encoding.side_effect = ImportError("No module named 'tiktoken'")

        count = count_tokens("Hello world", model=None)
        # Should fall back to approximate count (~4 chars per token)
        assert count > 0


def test_count_message_tokens_with_model():
    """Should accept model parameter for message token counting."""
    messages = [
        {"role": "user", "content": "Hello"},
    ]
    count = count_message_tokens(messages, model="gpt-4o")
    assert count > 0


def test_count_message_tokens_missing_content():
    """Should handle messages with missing content key."""
    messages = [
        {"role": "user"},  # No content
    ]
    count = count_message_tokens(messages)
    assert count >= 0  # Should not crash


def test_count_message_tokens_missing_role():
    """Should handle messages with missing role key."""
    messages = [
        {"content": "Hello"},  # No role
    ]
    count = count_message_tokens(messages)
    assert count >= 0  # Should not crash
