"""Tests for the pricing utility."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from llmsevals.utils.pricing import (
    _warn_if_stale,
    estimate_cost,
    get_model_pricing,
    MODEL_PRICING,
    PRICING_LAST_UPDATED,
    update_pricing,
)


def test_get_pricing_known_model():
    """Should return pricing for known models."""
    pricing = get_model_pricing("gpt-4o")
    assert pricing is not None
    assert len(pricing) == 2
    assert pricing[0] > 0  # input price
    assert pricing[1] > 0  # output price


def test_get_pricing_alias():
    """Should resolve model aliases."""
    pricing = get_model_pricing("gpt4o")
    assert pricing is not None

    pricing2 = get_model_pricing("claude-3-opus")
    assert pricing2 is not None


def test_get_pricing_unknown():
    """Should return None for unknown models."""
    pricing = get_model_pricing("totally-fake-model-xyz")
    assert pricing is None


def test_get_pricing_partial_match():
    """Should partial-match versioned model names."""
    # "gpt-4o" should match if we pass a longer version string
    pricing = get_model_pricing("gpt-4o-2024-08-06")
    assert pricing is not None


def test_estimate_cost_gpt4o():
    """Should calculate cost correctly for GPT-4o."""
    cost = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    assert cost is not None
    assert cost > 0
    # GPT-4o: $0.0025/1k input + $0.01/1k output
    # Expected: (1000/1000 * 0.0025) + (500/1000 * 0.01) = 0.0025 + 0.005 = 0.0075
    assert abs(cost - 0.0075) < 0.0001


def test_estimate_cost_gpt4o_mini():
    """Should be much cheaper than GPT-4o."""
    cost_mini = estimate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    cost_4o = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    assert cost_mini is not None
    assert cost_4o is not None
    assert cost_mini < cost_4o


def test_estimate_cost_unknown():
    """Should return None for unknown models."""
    cost = estimate_cost("fake-model", prompt_tokens=100, completion_tokens=100)
    assert cost is None


def test_estimate_cost_zero_tokens():
    """Should return 0 for zero tokens."""
    cost = estimate_cost("gpt-4o", prompt_tokens=0, completion_tokens=0)
    assert cost == 0.0


def test_model_pricing_has_major_models():
    """Pricing database should contain all major models."""
    major_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
    for model in major_models:
        assert model in MODEL_PRICING, f"Missing pricing for {model}"


def test_pricing_all_positive():
    """All pricing values should be positive."""
    for model, (input_price, output_price) in MODEL_PRICING.items():
        assert input_price > 0, f"{model} input price should be positive"
        assert output_price > 0, f"{model} output price should be positive"


# =============================================================================
# Missing coverage tests for 95%+ coverage
# =============================================================================


def test_update_pricing_adds_new_model():
    """Should add new model pricing to the database."""
    test_model = "test-custom-model-xyz"
    # Ensure model doesn't exist initially
    assert test_model not in MODEL_PRICING

    update_pricing(test_model, input_per_1k=0.001, output_per_1k=0.002)

    assert test_model in MODEL_PRICING
    assert MODEL_PRICING[test_model] == (0.001, 0.002)

    # Cleanup
    del MODEL_PRICING[test_model]


def test_update_pricing_overrides_existing():
    """Should override existing model pricing."""
    test_model = "test-override-model"
    MODEL_PRICING[test_model] = (0.01, 0.02)

    update_pricing(test_model, input_per_1k=0.005, output_per_1k=0.010)

    assert MODEL_PRICING[test_model] == (0.005, 0.010)

    # Cleanup
    del MODEL_PRICING[test_model]


def test_warn_if_stale_skipped_with_env_var():
    """Should skip warning when LLMSEVALS_SKIP_PRICING_WARNING is set."""
    with patch.dict(os.environ, {"LLMSEVALS_SKIP_PRICING_WARNING": "1"}):
        # Clear cache to ensure fresh call
        _warn_if_stale.cache_clear()
        # Should not raise or log anything
        result = _warn_if_stale()
        assert result is None


def test_warn_if_stale_logs_warning_when_stale():
    """Should log warning when pricing data is old."""
    from datetime import datetime

    # Mock an old date (100 days ago)
    old_date = (date.today() - timedelta(days=100)).strftime("%Y-%m-%d")

    with patch("llmsevals.utils.pricing.PRICING_LAST_UPDATED", old_date):
        # Re-patch the function to test actual logic
        # We need to test the actual function with mocked dependencies
        from llmsevals.utils import pricing

        # Clear cache
        pricing._warn_if_stale.cache_clear()

        # Mock the logger and datetime
        with patch("llmsevals.utils.pricing.datetime") as mock_datetime:
            mock_datetime.strptime = datetime.strptime
            mock_datetime.today.return_value = date(2026, 8, 7)  # 100 days after April 29

            with patch("llmsevals.utils.logger.get_logger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                # Call the function
                pricing._warn_if_stale()

                # Should have logged a warning
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "days old" in warning_msg
                assert "Pricing data is" in warning_msg


def test_get_pricing_partial_match_best_key():
    """Should select the longest matching key for partial matches."""
    # Test case: "gpt-4o-2024-11-20" should match "gpt-4o" not "gpt-4"
    # "gpt-4o" is longer than "gpt-4", so it should win
    from llmsevals.utils.pricing import MODEL_PRICING

    # Verify we have both keys for the test to be meaningful
    assert "gpt-4o" in MODEL_PRICING
    assert "gpt-4" in MODEL_PRICING

    pricing = get_model_pricing("gpt-4o-2024-11-20")
    # Should match gpt-4o pricing, not gpt-4
    assert pricing is not None
    assert pricing == MODEL_PRICING["gpt-4o"]
