"""Model pricing data for cost estimation.

Prices are in USD per 1,000 tokens. Updated as of April 2026.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from functools import lru_cache

PRICING_LAST_UPDATED = "2026-04-29"
_STALENESS_DAYS = 90


@lru_cache(maxsize=1)
def _warn_if_stale() -> None:
    """Warn once per process if pricing data is over 90 days old."""
    if os.environ.get("LLMSEVALS_SKIP_PRICING_WARNING"):
        return
    last = datetime.strptime(PRICING_LAST_UPDATED, "%Y-%m-%d").date()
    age = (date.today() - last).days
    if age > _STALENESS_DAYS:
        from llmsevals.utils.logger import get_logger as _gl
        _gl("llmsevals.pricing").warning(
            f"Pricing data is {age} days old (last updated: {PRICING_LAST_UPDATED}). "
            "Cost estimates may be inaccurate. "
            "Override: pricing.update_pricing(model, input_per_1k, output_per_1k). "
            "Suppress: set LLMSEVALS_SKIP_PRICING_WARNING=1"
        )


# Pricing: (input_price_per_1k_tokens, output_price_per_1k_tokens)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "o1": (0.015, 0.06),
    "o1-mini": (0.003, 0.012),
    "o3-mini": (0.0011, 0.0044),
    # Anthropic
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-sonnet-20240229": (0.003, 0.015),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "claude-3.5-sonnet": (0.003, 0.015),
    "claude-3.5-haiku": (0.0008, 0.004),
    "claude-opus-4-20250514": (0.015, 0.075),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    # Google
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-2.0-flash": (0.0001, 0.0004),
    "gemini-2.5-pro": (0.00125, 0.01),
    "gemini-2.5-flash": (0.000075, 0.0003),
    # Meta (via API providers)
    "llama-3.1-70b": (0.00035, 0.0004),
    "llama-3.1-405b": (0.001, 0.001),
    "llama-3.3-70b": (0.00035, 0.0004),
}

# Aliases for common shorthand names
MODEL_ALIASES: dict[str, str] = {
    "gpt4o": "gpt-4o",
    "gpt4": "gpt-4",
    "gpt-4o-2024-08-06": "gpt-4o",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "gemini-pro": "gemini-1.5-pro",
    "gemini-flash": "gemini-1.5-flash",
}


def get_model_pricing(model: str) -> tuple[float, float] | None:
    """Get pricing for a model.

    Args:
        model: Model identifier (e.g. "gpt-4o", "claude-3-opus").

    Returns:
        Tuple of (input_price_per_1k, output_price_per_1k) or None if unknown.
    """
    # Check direct match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Check aliases
    resolved = MODEL_ALIASES.get(model)
    if resolved and resolved in MODEL_PRICING:
        return MODEL_PRICING[resolved]

    # Partial match (for versioned model names like "gpt-4o-2024-11-20")
    # Only match if the known key is a prefix of the requested model name,
    # which avoids "o3" matching "o3-mini" or "gpt-4" matching "gpt-4o".
    model_lower = model.lower()
    best_match: tuple[float, float] | None = None
    best_key_len = 0
    for key, pricing in MODEL_PRICING.items():
        if model_lower.startswith(key) and len(key) > best_key_len:
            best_match = pricing
            best_key_len = len(key)

    return best_match


def update_pricing(model: str, input_per_1k: float, output_per_1k: float) -> None:
    """Override or add model pricing at runtime.

    Args:
        model: Model identifier (e.g. "my-custom-model").
        input_per_1k: Price per 1,000 input tokens in USD.
        output_per_1k: Price per 1,000 output tokens in USD.

    Example::

        from llmsevals.utils import pricing
        pricing.update_pricing("my-model", input_per_1k=0.001, output_per_1k=0.002)
    """
    MODEL_PRICING[model] = (input_per_1k, output_per_1k)


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Estimate the cost of an LLM call.

    Args:
        model: Model identifier.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD, or None if pricing is unknown.
    """
    _warn_if_stale()
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    input_price, output_price = pricing
    cost = (prompt_tokens / 1000 * input_price) + (completion_tokens / 1000 * output_price)
    return round(cost, 8)
