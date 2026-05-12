"""Tests for the OpenAI provider (fully mocked — no API key needed)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmsevals.providers.openai_provider import OpenAIProvider
from llmsevals.core.base_provider import GenerationResult


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI chat completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Paris is the capital of France."
    response.model = "gpt-4o-mini"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 12
    response.usage.completion_tokens = 8
    response.usage.total_tokens = 20
    return response


@pytest.fixture
def mock_openai_response_no_usage():
    """Create a mock OpenAI response without usage data."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Hello"
    response.model = "gpt-4o-mini"
    response.usage = None
    return response


def test_openai_provider_init():
    """Should initialize with default parameters."""
    provider = OpenAIProvider()
    assert provider.model == "gpt-4o-mini"
    assert provider.temperature == 0.0
    assert provider.max_tokens == 1024


def test_openai_provider_custom_init():
    """Should initialize with custom parameters."""
    provider = OpenAIProvider(
        model="gpt-4o",
        api_key="sk-test-key",
        base_url="https://custom.api.com",
        temperature=0.5,
        max_tokens=2048,
    )
    assert provider.model == "gpt-4o"
    assert provider.api_key == "sk-test-key"
    assert provider.base_url == "https://custom.api.com"
    assert provider.temperature == 0.5
    assert provider.max_tokens == 2048


def test_openai_provider_repr():
    """Should have a useful repr."""
    provider = OpenAIProvider(model="gpt-4o")
    assert "OpenAIProvider" in repr(provider)
    assert "gpt-4o" in repr(provider)


@pytest.mark.asyncio
async def test_openai_generate(mock_openai_response):
    """Should generate a response via the mocked client."""
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")

    # Mock the client
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
    provider._client = mock_client

    result = await provider.generate("What is the capital of France?")

    assert isinstance(result, GenerationResult)
    assert result.text == "Paris is the capital of France."
    assert result.model == "gpt-4o-mini"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 8
    assert result.total_tokens == 20
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_openai_generate_with_messages(mock_openai_response):
    """Should generate from a message list."""
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
    provider._client = mock_client

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    result = await provider.generate_with_messages(messages)

    assert result.text == "Paris is the capital of France."
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_openai_generate_no_usage(mock_openai_response_no_usage):
    """Should handle response with no usage data."""
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response_no_usage)
    provider._client = mock_client

    result = await provider.generate("Hello")

    assert result.text == "Hello"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0


@pytest.mark.asyncio
async def test_openai_generate_api_error():
    """Should propagate API errors."""
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
    provider._client = mock_client

    with pytest.raises(Exception, match="API Error"):
        await provider.generate("test")


@pytest.mark.asyncio
async def test_openai_generate_empty_content():
    """Should handle None content in response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = None
    response.model = "gpt-4o-mini"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 5
    response.usage.completion_tokens = 0
    response.usage.total_tokens = 5

    provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)
    provider._client = mock_client

    result = await provider.generate("test")
    assert result.text == ""


def test_openai_client_lazy_init():
    """Client should be lazy-initialized with correct kwargs."""
    import sys

    provider = OpenAIProvider(
        model="gpt-4o",
        api_key="sk-test-key",
        base_url="https://custom.api.com",
    )
    assert provider._client is None

    # Create a fake openai module with AsyncOpenAI
    mock_async_openai = MagicMock()
    mock_module = MagicMock()
    mock_module.AsyncOpenAI = mock_async_openai

    with patch.dict(sys.modules, {"openai": mock_module}):
        # Reset client so property re-runs
        provider._client = None
        client = provider.client

        mock_async_openai.assert_called_once_with(
            api_key="sk-test-key",
            base_url="https://custom.api.com",
        )
        assert client is not None


def test_openai_client_no_api_key():
    """Client should work without explicit api_key."""
    import sys

    provider = OpenAIProvider(model="gpt-4o")
    provider._api_key = None
    provider.base_url = None

    mock_async_openai = MagicMock()
    mock_module = MagicMock()
    mock_module.AsyncOpenAI = mock_async_openai

    with patch.dict(sys.modules, {"openai": mock_module}):
        provider._client = None
        provider.client
        mock_async_openai.assert_called_once_with()


def test_openai_client_cached():
    """Client should be cached after first access."""
    import sys

    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test")

    mock_instance = MagicMock()
    mock_async_openai = MagicMock(return_value=mock_instance)
    mock_module = MagicMock()
    mock_module.AsyncOpenAI = mock_async_openai

    with patch.dict(sys.modules, {"openai": mock_module}):
        provider._client = None
        client1 = provider.client
        client2 = provider.client

        assert client1 is client2
        mock_async_openai.assert_called_once()  # Only created once
