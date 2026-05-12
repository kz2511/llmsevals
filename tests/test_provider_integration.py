"""Provider integration tests for real-world API compatibility.

Tests provider implementations with realistic scenarios and error handling.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmsevals.core.base_provider import BaseProvider, GenerationResult
from llmsevals.providers.openai_provider import OpenAIProvider


# =============================================================================
# OpenAI Provider Real Integration Tests
# =============================================================================


@pytest.mark.real_world
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping real API test"
)
@pytest.mark.asyncio
async def test_openai_provider_real_api():
    """Test with real OpenAI API (requires valid API key)."""
    provider = OpenAIProvider(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=100
    )

    try:
        result = await provider.generate("What is 2+2? Answer with just the number.")

        assert result.text is not None
        assert len(result.text) > 0
        assert result.latency_ms > 0
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0
        assert result.total_tokens > 0
        assert result.model == "gpt-4o-mini"

    finally:
        await provider.close()


@pytest.mark.real_world
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping real API test"
)
@pytest.mark.asyncio
async def test_openai_provider_chat_completion():
    """Test chat completion with messages format."""
    provider = OpenAIProvider(model="gpt-4o-mini", temperature=0.0)

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'hello' and nothing else."},
        ]

        result = await provider.generate_with_messages(messages)

        assert result.text is not None
        assert "hello" in result.text.lower()
        assert result.model == "gpt-4o-mini"

    finally:
        await provider.close()


@pytest.mark.real_world
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping real API test"
)
@pytest.mark.asyncio
async def test_openai_provider_error_handling():
    """Test error handling with real API (invalid model)."""
    provider = OpenAIProvider(model="invalid-model-name-xyz123")

    try:
        with pytest.raises(Exception):  # Should raise an API error
            await provider.generate("Test prompt")
    finally:
        await provider.close()


# =============================================================================
# Mock Provider Tests (for CI/CD without API keys)
# =============================================================================


class AsyncMockProvider(BaseProvider):
    """Configurable async mock provider for testing."""

    def __init__(
        self,
        response_delay_ms: float = 0,
        fail_after_n_calls: int | None = None,
        custom_responses: dict[str, str] | None = None,
    ):
        super().__init__(model="async-mock-v1")
        self.response_delay_ms = response_delay_ms
        self.fail_after_n_calls = fail_after_n_calls
        self.custom_responses = custom_responses or {}
        self.call_count = 0

    async def generate(self, prompt: str) -> GenerationResult:
        """Generate with configurable behavior."""
        import asyncio

        self.call_count += 1

        if self.fail_after_n_calls and self.call_count >= self.fail_after_n_calls:
            raise Exception(f"Simulated failure after {self.fail_after_n_calls} calls")

        if self.response_delay_ms > 0:
            await asyncio.sleep(self.response_delay_ms / 1000)

        # Use custom response if available
        response_text = self.custom_responses.get(prompt, f"Response to: {prompt[:30]}")

        tokens = len(response_text.split()) * 2

        return GenerationResult(
            text=response_text,
            latency_ms=self.response_delay_ms,
            prompt_tokens=len(prompt.split()),
            completion_tokens=tokens,
            total_tokens=len(prompt.split()) + tokens,
            model=self.model,
        )

    async def generate_with_messages(self, messages: list[dict[str, str]]) -> GenerationResult:
        """Generate from messages."""
        last_content = messages[-1].get("content", "") if messages else ""
        return await self.generate(last_content)


@pytest.mark.asyncio
async def test_async_mock_provider_basic():
    """Test async mock provider generates responses."""
    provider = AsyncMockProvider(
        custom_responses={"Hello": "World!"}
    )

    result = await provider.generate("Hello")
    assert result.text == "World!"
    assert result.model == "async-mock-v1"


@pytest.mark.asyncio
async def test_async_mock_provider_failure_mode():
    """Test async mock provider failure simulation."""
    provider = AsyncMockProvider(fail_after_n_calls=2)

    # First call succeeds
    result1 = await provider.generate("Test 1")
    assert result1.text is not None

    # Second call fails
    with pytest.raises(Exception, match="Simulated failure"):
        await provider.generate("Test 2")


@pytest.mark.asyncio
async def test_async_mock_provider_delay():
    """Test async mock provider with delay."""
    import time

    provider = AsyncMockProvider(response_delay_ms=50)  # 50ms delay

    start = time.time()
    result = await provider.generate("Test")
    elapsed = (time.time() - start) * 1000

    assert elapsed >= 45  # Should have waited ~50ms (with some tolerance)
    assert result.latency_ms == 50.0


# =============================================================================
# Provider Configuration Tests
# =============================================================================


def test_openai_provider_api_key_security():
    """Verify API key is not exposed in string representations."""
    provider = OpenAIProvider(
        model="gpt-4o",
        api_key="sk-secret-key-12345",
    )

    repr_str = repr(provider)
    str_str = str(provider)

    assert "secret" not in repr_str.lower()
    assert "sk-" not in repr_str
    assert "12345" not in repr_str


def test_openai_provider_from_env_var(monkeypatch):
    """Test API key loading from environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-67890")

    provider = OpenAIProvider(model="gpt-4o")
    assert provider._api_key == "sk-env-key-67890"


def test_openai_provider_custom_base_url():
    """Test custom base URL configuration (for Azure/proxy)."""
    provider = OpenAIProvider(
        model="gpt-4o",
        base_url="https://custom.openai.api.com/v1",
        api_key="test-key",
    )

    assert provider.base_url == "https://custom.openai.api.com/v1"


def test_openai_provider_temperature_and_max_tokens():
    """Test temperature and max_tokens configuration."""
    provider = OpenAIProvider(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=500,
    )

    assert provider.temperature == 0.7
    assert provider.max_tokens == 500


# =============================================================================
# Provider Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_openai_provider_empty_choices():
    """Test handling of empty choices in API response."""
    # Create provider with dummy key to avoid initialization error
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")

    mock_response = MagicMock()
    mock_response.choices = []
    mock_response.model = "gpt-4o-mini"

    # Mock the client directly
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    provider._client = mock_client

    with pytest.raises(ValueError, match="empty choices"):
        await provider.generate("Test prompt")


@pytest.mark.asyncio
async def test_openai_provider_api_error_logging():
    """Test that API errors are properly logged."""
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")

    # Mock the client directly
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("API Error: Rate limit exceeded")
    )
    provider._client = mock_client

    with pytest.raises(Exception, match="API Error"):
        await provider.generate("Test")


@pytest.mark.asyncio
async def test_openai_provider_close_idempotent():
    """Test that close() can be called multiple times safely."""
    provider = OpenAIProvider(model="gpt-4o-mini")

    # Close when client is None (should be safe)
    await provider.close()

    # Simulate client creation
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    provider._client = mock_client

    # First close
    await provider.close()
    assert provider._client is None

    # Second close should not raise
    await provider.close()


# =============================================================================
# Provider Comparison/Compatibility Tests
# =============================================================================


def test_provider_interface_compliance():
    """Verify all providers implement required interface."""
    from llmsevals.core.base_provider import BaseProvider

    # OpenAIProvider should inherit from BaseProvider
    assert issubclass(OpenAIProvider, BaseProvider)

    # Required methods should exist
    assert hasattr(OpenAIProvider, "generate")
    assert hasattr(OpenAIProvider, "generate_with_messages")
    assert hasattr(OpenAIProvider, "close")


@pytest.mark.asyncio
async def test_provider_generation_result_structure():
    """Verify GenerationResult has all required fields."""
    provider = AsyncMockProvider()

    result = await provider.generate("Test prompt")

    # Required fields
    assert hasattr(result, "text")
    assert hasattr(result, "latency_ms")
    assert hasattr(result, "prompt_tokens")
    assert hasattr(result, "completion_tokens")
    assert hasattr(result, "total_tokens")
    assert hasattr(result, "model")
    assert hasattr(result, "raw_response")

    # Type checks
    assert isinstance(result.text, str)
    assert isinstance(result.latency_ms, (int, float))
    assert isinstance(result.prompt_tokens, int)
    assert isinstance(result.completion_tokens, int)
    assert isinstance(result.total_tokens, int)
    assert isinstance(result.model, str)


# =============================================================================
# Multi-provider and Fallback Tests
# =============================================================================


@pytest.mark.asyncio
async def test_multiple_provider_instances():
    """Test that multiple provider instances work independently."""
    provider1 = AsyncMockProvider(custom_responses={"test": "response1"})
    provider2 = AsyncMockProvider(custom_responses={"test": "response2"})

    result1 = await provider1.generate("test")
    result2 = await provider2.generate("test")

    assert result1.text == "response1"
    assert result2.text == "response2"
    assert provider1.call_count == 1
    assert provider2.call_count == 1

