"""Input sanitization and truncation utilities for llmsevals.

Provides defenses against prompt injection and utilities for
managing token budgets in evaluation prompts.
"""

from __future__ import annotations

import re

from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)

# Compiled patterns for common prompt injection phrases.
# Each pattern is replaced with "[REDACTED]" in user-supplied text.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # Tag-style injection: <instruction>, <system>, <override>, etc.
    re.compile(
        r"&lt;/?(?:instruction|system|human|assistant|prompt|hidden|override|ignore)[^&]{0,60}&gt;",
        re.IGNORECASE,
    ),
    # "Ignore / disregard previous instructions" variants
    re.compile(
        r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|the\s+above)\s+instructions?\b",
        re.IGNORECASE,
    ),
    # "You are now a ..." persona hijack
    re.compile(r"\byou\s+are\s+now\s+(?:a|an|the)\s+\w+", re.IGNORECASE),
    # "Act as a ..." persona hijack
    re.compile(r"\bact\s+as\s+(?:a|an|the)\s+\w+", re.IGNORECASE),
    # Explicit score/rating override attempts (e.g. "Score: 1.0", "rating=0")
    re.compile(r"\b(?:score|rating)\s*[=:]\s*[01](?:\.\d+)?\b", re.IGNORECASE),
    # Chat-style role delimiters that could hijack message structure
    re.compile(
        r"(?:^|\n)\s*(?:system|user|assistant|human)\s*:\s*",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def sanitize_input(text: str) -> str:
    """Sanitize user-provided text to mitigate prompt injection.

    Two-stage defense:
    1. Escape all ``<`` and ``>`` characters so no XML/HTML tag structure
       can survive into the judge prompt (tag-based injection neutralised).
    2. Redact known injection phrases (persona hijack, instruction override,
       score manipulation, role delimiter injection).

    This is a defense-in-depth measure — it does NOT guarantee full
    protection against every sophisticated attack.

    Args:
        text: Raw user text (e.g., actual_output, context, input).

    Returns:
        Sanitized text safe to embed inside ``<user_data>`` prompt fences.
    """
    if not text:
        return text

    # Stage 1 — neutralise all XML/HTML angle brackets so no tag survives.
    # Must happen BEFORE the injection-pattern scan so that patterns that
    # match the escaped form (e.g. &lt;instruction&gt;) are also caught.
    text = text.replace("<", "&lt;").replace(">", "&gt;")

    # Stage 2 — redact known injection phrases
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub("[REDACTED]", text)

    return text


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    model: str | None = None,
) -> str:
    """Truncate text to fit within a token budget.

    Uses tiktoken for accurate counting, falls back to character-based
    approximation if tiktoken is unavailable.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens allowed.
        model: Optional model name for encoding selection.

    Returns:
        Truncated text that fits within the token budget.
        If already within budget, returns text unchanged.
    """
    if not text:
        return text

    from llmsevals.utils.tokenizer import count_tokens

    current_tokens = count_tokens(text, model)

    if current_tokens <= max_tokens:
        return text

    # Binary search for the right truncation point
    low, high = 0, len(text)
    best = 0

    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        token_count = count_tokens(candidate, model)

        if token_count <= max_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    truncated = text[:best]
    logger.info(
        f"Truncated text from {current_tokens} to ~{max_tokens} tokens "
        f"({len(text)} -> {len(truncated)} chars)"
    )

    return truncated
