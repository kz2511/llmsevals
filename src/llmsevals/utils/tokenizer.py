"""Token counting utilities using tiktoken."""

from __future__ import annotations

from functools import lru_cache

from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)

# Default encoding for token counting
_DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=16)
def _get_encoding(key: str, use_model: bool = True) -> object:
    """Cached tiktoken encoding lookup — avoids re-parsing encoding files per call."""
    import tiktoken

    if use_model:
        try:
            return tiktoken.encoding_for_model(key)
        except KeyError:
            return tiktoken.get_encoding(_DEFAULT_ENCODING)
    return tiktoken.get_encoding(key)


def clear_encoding_cache() -> None:
    """Clear the tiktoken encoding cache. Useful in tests and multiprocess resets."""
    _get_encoding.cache_clear()


def count_tokens(text: str, model: str | None = None) -> int:
    """Count the number of tokens in a text string.

    Uses tiktoken for accurate token counting. Falls back to a simple
    word-based approximation if tiktoken is not available.

    Args:
        text: The text to count tokens for.
        model: Optional model name to select the correct encoding.
            Defaults to cl100k_base (GPT-4 / GPT-3.5 encoding).

    Returns:
        The number of tokens in the text.
    """
    try:
        if model:
            encoding = _get_encoding(model, use_model=True)
        else:
            encoding = _get_encoding(_DEFAULT_ENCODING, use_model=False)

        return len(encoding.encode(text))  # type: ignore[union-attr]
    except ImportError:
        logger.warning(
            "tiktoken not installed, using approximate token count. "
            "Install with: pip install tiktoken"
        )
        # Rough approximation: ~4 characters per token, minimum 1 for non-empty text
        return max(1, len(text) // 4) if text else 0


def count_message_tokens(
    messages: list[dict[str, str]], model: str | None = None
) -> int:
    """Count tokens for a list of chat messages.

    Accounts for the overhead tokens per message that OpenAI-style
    chat APIs use.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        model: Optional model name for encoding selection.

    Returns:
        Total token count including message overhead.
    """
    # Per-message overhead (role tokens, separators, etc.)
    overhead_per_message = 4

    total = 0
    for msg in messages:
        total += overhead_per_message
        total += count_tokens(msg.get("content", ""), model)
        total += count_tokens(msg.get("role", ""), model)

    # Final assistant reply priming
    total += 2
    return total
