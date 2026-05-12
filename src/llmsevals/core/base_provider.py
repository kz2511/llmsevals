"""BaseProvider — Abstract interface for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    """Result from an LLM generation call.

    Attributes:
        text: The generated text response.
        latency_ms: Time taken for the API call in milliseconds.
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used.
        model: The model identifier used.
        raw_response: The raw response object from the provider.
            WARNING: Do not serialize or log this field — it may contain
            sensitive headers or authentication tokens.
    """

    text: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    raw_response: Any = field(default=None, repr=False)

    @property
    def token_count(self) -> dict[str, int]:
        """Token usage as a dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize to dict, explicitly excluding raw_response."""
        return {
            "text": self.text,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
        }


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must support:
        - ``generate(prompt)``: Generate a response from a text prompt.
        - ``generate_with_messages(messages)``: Generate from a message list.

    Args:
        model: The model identifier (e.g. "gpt-4o", "claude-3-opus").
        api_key: API key for the provider. If None, reads from environment.
        base_url: Custom base URL for the API endpoint.
        temperature: Sampling temperature (0.0–2.0). Defaults to 0.0 for
            deterministic evaluation.
        max_tokens: Maximum tokens to generate.

    Example::

        from llmsevals.providers import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o")
        result = await provider.generate("What is 2 + 2?")
        print(result.text)        # "4"
        print(result.latency_ms)  # 234.5
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self._api_key = api_key  # Private: never expose directly
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def api_key(self) -> str | None:
        """Read-only access to api_key. Kept for backwards compatibility."""
        return self._api_key

    @property
    def has_api_key(self) -> bool:
        """True if an API key is configured."""
        return bool(self._api_key)

    @abstractmethod
    async def generate(self, prompt: str) -> GenerationResult:
        """Generate a response from a text prompt.

        Args:
            prompt: The text prompt to send to the LLM.

        Returns:
            A GenerationResult with text, latency, and token usage.
        """
        ...

    @abstractmethod
    async def generate_with_messages(
        self, messages: list[dict[str, str]]
    ) -> GenerationResult:
        """Generate a response from a list of chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            A GenerationResult with text, latency, and token usage.
        """
        ...

    async def close(self) -> None:
        """Close the provider and release resources.

        Override in subclasses that hold HTTP clients or connections.
        """

    async def __aenter__(self) -> BaseProvider:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        key_display = "***" if self._api_key else "None"
        return f"{self.__class__.__name__}(model='{self.model}', api_key='{key_display}')"
