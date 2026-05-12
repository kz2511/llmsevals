"""LLM-as-a-Judge — Uses an LLM to evaluate another LLM's output."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from llmsevals.core.base_provider import BaseProvider
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional OpenAI-specific error types for fine-grained retry decisions.
# If openai is not installed we fall back to treating all exceptions as
# generic transient errors (retry with standard backoff).
# ---------------------------------------------------------------------------
try:
    from openai import APIConnectionError as _APIConnectionError
    from openai import APIStatusError as _APIStatusError
    from openai import APITimeoutError as _APITimeoutError
    from openai import RateLimitError as _RateLimitError
    _OPENAI_ERRORS_AVAILABLE = True
except ImportError:
    # Sentinel: these will never be raised at runtime when openai is absent,
    # but give isinstance() a valid type so the type checker is satisfied.
    class _RateLimitError(Exception): pass      # type: ignore[no-redef]
    class _APIConnectionError(Exception): pass  # type: ignore[no-redef]
    class _APITimeoutError(Exception): pass     # type: ignore[no-redef]
    class _APIStatusError(Exception):           # type: ignore[no-redef]
        status_code: int = 0
    _OPENAI_ERRORS_AVAILABLE = False


class LLMJudge:
    """Orchestrates LLM-as-a-Judge evaluation.

    Uses a configured LLM provider to evaluate outputs by sending
    structured prompts and parsing JSON scores from the response.

    Args:
        provider: An LLM provider instance to use as the judge.
            If None, will attempt to auto-create an OpenAI provider.
        score_scale: The maximum score the judge is expected to return.
            Scores are normalized to [0, 1] by dividing by this value.
            Defaults to 1.0 (judge returns 0.0–1.0 directly).
            Set to 10.0 if the judge uses a 0–10 scale.
        max_retries: Maximum number of retries on transient failures.
        retry_delay: Base delay in seconds between retries (exponential backoff).

    Example::

        from llmsevals.judge import LLMJudge
        from llmsevals.providers import OpenAIProvider

        judge = LLMJudge(provider=OpenAIProvider(model="gpt-4o-mini"))
        result = await judge.evaluate(
            prompt_template="Rate this: {text}",
            variables={"text": "Hello world"},
        )
        print(result)  # {"score": 0.8, "reason": "..."}
    """

    def __init__(
        self,
        provider: BaseProvider | None = None,
        score_scale: float = 1.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._provider = provider
        self.score_scale = score_scale
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @property
    def provider(self) -> BaseProvider:
        """Get or auto-create the judge provider."""
        if self._provider is None:
            self._provider = self._create_default_provider()
        return self._provider

    def _create_default_provider(self) -> BaseProvider:
        """Create a default OpenAI provider for judging.

        Model resolution order:
        1. ``LLMSEVALS_JUDGE_MODEL`` env var
        2. ``LLMSEVALS_OPENAI_MODEL`` env var
        3. Hard fallback: ``gpt-4o-mini``
        """
        import os

        try:
            from llmsevals.providers.openai_provider import OpenAIProvider

            judge_model = (
                os.environ.get("LLMSEVALS_JUDGE_MODEL")
                or os.environ.get("LLMSEVALS_OPENAI_MODEL")
                or "gpt-4o-mini"
            )
            return OpenAIProvider(model=judge_model, temperature=0.0)
        except ImportError:
            raise ImportError(
                "No judge provider configured and OpenAI is not installed. "
                "Install with: pip install llmsevals[openai]"
            )

    # Maximum total tokens across all variable values before truncation kicks in.
    # Keeps judge prompts safely within gpt-4o-mini's 128k context window while
    # leaving headroom for the prompt template, preamble, and response tokens.
    MAX_INPUT_TOKENS: int = 8000

    async def evaluate(
        self,
        prompt_template: str,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        """Send an evaluation prompt to the judge LLM and parse the response.

        Args:
            prompt_template: A prompt template with {variable} placeholders.
            variables: Dict of variables to substitute into the template.

        Returns:
            A dict with 'score' (float) and 'reason' (str) keys.
        """
        from llmsevals.utils.sanitizer import sanitize_input, truncate_to_tokens
        from llmsevals.utils.tokenizer import count_tokens

        # QUAL-02: Guard against inputs that exceed the judge's context window.
        # Measure total tokens across all variable values and truncate proportionally
        # if the combined size exceeds MAX_INPUT_TOKENS.
        str_vars: dict[str, str] = {k: str(v) for k, v in variables.items()}
        total_tokens = sum(count_tokens(v) for v in str_vars.values())
        if total_tokens > self.MAX_INPUT_TOKENS:
            logger.warning(
                f"Judge input too large ({total_tokens} tokens across "
                f"{len(str_vars)} variables, limit {self.MAX_INPUT_TOKENS}). "
                "Truncating proportionally."
            )
            budget_per_var = self.MAX_INPUT_TOKENS // max(len(str_vars), 1)
            str_vars = {
                k: truncate_to_tokens(v, budget_per_var) if count_tokens(v) > budget_per_var else v
                for k, v in str_vars.items()
            }

        # Sanitize variable values to prevent prompt injection
        sanitized_vars = {k: sanitize_input(v) for k, v in str_vars.items()}

        # Format the prompt
        prompt = prompt_template.format(**sanitized_vars)

        # Call the judge LLM with rate-limit-aware retry logic (SCALE-04).
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                result = await self.provider.generate(prompt)
                parsed = self._parse_response(result.text)
                return parsed
            except Exception as e:
                last_error = e
                if attempt >= self.max_retries - 1:
                    break

                # Decide retry strategy based on error type.
                if _OPENAI_ERRORS_AVAILABLE:
                    if isinstance(e, _RateLimitError):
                        # Rate-limited: use aggressive backoff (4^attempt)
                        delay = self.retry_delay * (4 ** attempt)
                        logger.warning(
                            f"Judge rate-limited on attempt {attempt + 1}/"
                            f"{self.max_retries}. Retrying in {delay:.1f}s..."
                        )
                    elif isinstance(e, (_APIConnectionError, _APITimeoutError)):
                        # Transient network error: standard backoff
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Judge network error ({type(e).__name__}) on attempt "
                            f"{attempt + 1}/{self.max_retries}. Retrying in {delay:.1f}s..."
                        )
                    elif isinstance(e, _APIStatusError) and e.status_code >= 500:
                        # Server-side error: standard backoff
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Judge server error (HTTP {e.status_code}) on attempt "
                            f"{attempt + 1}/{self.max_retries}. Retrying in {delay:.1f}s..."
                        )
                    elif isinstance(e, _APIStatusError):
                        # Client-side error (4xx): non-retryable — fail immediately
                        logger.error(
                            f"Judge non-retryable client error (HTTP {e.status_code}). "
                            "Check model name, API key, and prompt format."
                        )
                        raise
                    else:
                        # Unknown error (parse failure etc.): standard backoff
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Judge evaluation attempt {attempt + 1}/{self.max_retries} "
                            f"failed ({type(e).__name__}). Retrying in {delay:.1f}s..."
                        )
                else:
                    # openai not installed — treat every error as transient
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Judge evaluation attempt {attempt + 1}/{self.max_retries} "
                        f"failed ({type(e).__name__}). Retrying in {delay:.1f}s..."
                    )

                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(
            f"Judge evaluation failed after {self.max_retries} attempts "
            f"({type(last_error).__name__})"
        )
        raise last_error  # type: ignore[misc]

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Parse the JSON response from the judge LLM.

        Handles common formatting issues like markdown code blocks,
        extra text before/after JSON, etc.

        Args:
            response_text: Raw text response from the judge.

        Returns:
            Parsed dict with 'score' and 'reason' keys.
        """
        text = response_text.strip()

        # Remove markdown code blocks if present
        if "```" in text:
            # Extract content between code blocks
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Try direct JSON parse
        try:
            data = json.loads(text)
            return self._validate_parsed(data)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text (supports nested objects)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_parsed(data)
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract score from text
        score_match = re.search(
            r"(?:score|rating)\s*(?:is|:|=)\s*(\d+\.?\d*)", text, re.IGNORECASE
        )
        if score_match:
            score = float(score_match.group(1))
            score = self._normalize_score(score)
            return {"score": score, "reason": text[:200]}

        logger.warning(f"Could not parse judge response: {text[:200]}")
        return {"score": 0.0, "reason": f"Failed to parse judge response: {text[:100]}"}

    def _validate_parsed(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize parsed judge response."""
        score = float(data.get("score", 0.0))
        reason = str(data.get("reason", ""))

        score = self._normalize_score(score)

        return {"score": score, "reason": reason}

    def _normalize_score(self, score: float) -> float:
        """Normalize a raw score to the [0, 1] range.

        Divides by ``score_scale`` and clamps to [0.0, 1.0].
        """
        if self.score_scale != 1.0 and self.score_scale > 0:
            score = score / self.score_scale

        # Clamp score to [0, 1]
        return min(max(score, 0.0), 1.0)
