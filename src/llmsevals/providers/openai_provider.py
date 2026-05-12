"""OpenAI provider implementation."""

from __future__ import annotations

import os
import time
from typing import Any

from llmsevals.core.base_provider import BaseProvider, GenerationResult
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI LLM provider.

    Supports all OpenAI models including GPT-4o, GPT-4, GPT-3.5, o1, o3, etc.

    Args:
        model: Model identifier (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
            If None, reads from ``LLMSEVALS_OPENAI_MODEL`` env var,
            then falls back to ``"gpt-4o-mini"``.
        api_key: OpenAI API key. If None, reads from ``OPENAI_API_KEY`` env var.
        base_url: Custom base URL (for Azure OpenAI or compatible APIs).
        temperature: Sampling temperature (0.0–2.0). Defaults to 0.0.
        max_tokens: Maximum tokens to generate. Defaults to 1024.

    Environment variables:
        ``LLMSEVALS_OPENAI_MODEL``: Default model when ``model`` is not passed.
        ``OPENAI_API_KEY``: OpenAI API key when ``api_key`` is not passed.

    Example::

        # Explicit model
        provider = OpenAIProvider(model="gpt-4o")

        # Model from env: export LLMSEVALS_OPENAI_MODEL=gpt-4o
        provider = OpenAIProvider()

        result = await provider.generate("What is 2 + 2?")
        print(result.text)  # "4"
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        resolved_model = (
            model
            or os.environ.get("LLMSEVALS_OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        super().__init__(
            model=resolved_model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.timeout = timeout
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-initialize the OpenAI async client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAIProvider. "
                    "Install with: pip install llmsevals[openai]"
                )

            kwargs: dict[str, Any] = {"timeout": self.timeout}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def generate(self, prompt: str) -> GenerationResult:
        """Generate a response from a text prompt.

        Args:
            prompt: The text prompt to send to OpenAI.

        Returns:
            A GenerationResult with text, latency, and token counts.
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_with_messages(messages)

    async def generate_with_messages(
        self, messages: list[dict[str, str]]
    ) -> GenerationResult:
        """Generate a response from a list of chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            A GenerationResult with text, latency, and token counts.
        """
        start_time = time.perf_counter()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        except Exception as e:
            # Log only the error type — full message may contain auth fragments.
            logger.error(
                "OpenAI API call failed",
                extra={"error_type": type(e).__name__, "model": self.model},
            )
            raise RuntimeError(
                f"OpenAI API call failed ({type(e).__name__}). "
                "Check OPENAI_API_KEY and network connectivity."
            ) from e

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract response data
        if not response.choices:
            raise ValueError(
                f"OpenAI returned an empty choices list for model '{self.model}'. "
                "This may be caused by a content filter or API error."
            )
        text = response.choices[0].message.content or ""
        usage = response.usage

        return GenerationResult(
            text=text,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=response.model or self.model,
            raw_response=response,
        )

    async def close(self) -> None:
        """Close the OpenAI async client and release resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None
