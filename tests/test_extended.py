"""Extended tests for judge, base_provider, eval_result, providers init, and utils."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmsevals.core.base_provider import BaseProvider, GenerationResult
from llmsevals.core.base_metric import MetricResult
from llmsevals.core.eval_result import EvalResult, TestCaseResult
from llmsevals.judge.judge import LLMJudge


# =============================================================================
# GenerationResult tests
# =============================================================================


def test_generation_result_token_count():
    """token_count property should return a dict."""
    gr = GenerationResult(
        text="Hello",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    tc = gr.token_count
    assert tc == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


def test_generation_result_defaults():
    """Should have sensible defaults."""
    gr = GenerationResult(text="Hi")
    assert gr.latency_ms == 0.0
    assert gr.prompt_tokens == 0
    assert gr.total_tokens == 0
    assert gr.model == ""
    assert gr.raw_response is None


# =============================================================================
# BaseProvider tests
# =============================================================================


class ConcreteProvider(BaseProvider):
    """Concrete provider for testing BaseProvider."""

    async def generate(self, prompt):
        return GenerationResult(text="response")

    async def generate_with_messages(self, messages):
        return GenerationResult(text="response")


def test_base_provider_init():
    """Should store all init params."""
    provider = ConcreteProvider(
        model="test-model",
        api_key="key",
        base_url="http://localhost",
        temperature=0.5,
        max_tokens=512,
    )
    assert provider.model == "test-model"
    assert provider.api_key == "key"
    assert provider.base_url == "http://localhost"
    assert provider.temperature == 0.5
    assert provider.max_tokens == 512


def test_base_provider_repr():
    """Should show class name and model."""
    provider = ConcreteProvider(model="test-model")
    r = repr(provider)
    assert "ConcreteProvider" in r
    assert "test-model" in r


@pytest.mark.asyncio
async def test_concrete_provider_generate():
    """Concrete provider should work."""
    provider = ConcreteProvider(model="test")
    result = await provider.generate("hello")
    assert result.text == "response"


@pytest.mark.asyncio
async def test_concrete_provider_generate_messages():
    """Concrete provider should handle messages."""
    provider = ConcreteProvider(model="test")
    result = await provider.generate_with_messages([{"role": "user", "content": "hi"}])
    assert result.text == "response"


# =============================================================================
# EvalResult extended tests
# =============================================================================


def test_testcase_result_empty_metrics_passed():
    """Empty metric_results should return passed=False (no evidence of passing)."""
    tcr = TestCaseResult(test_case_index=0, input="Q")
    assert tcr.passed is False


def test_testcase_result_empty_metrics_score():
    """Empty metric_results should return score=0.0."""
    tcr = TestCaseResult(test_case_index=0, input="Q")
    assert tcr.score == 0.0


def test_eval_result_summary_calls_reporter():
    """summary() should call ConsoleReporter.report()."""
    er = EvalResult(
        test_case_results=[
            TestCaseResult(
                test_case_index=0,
                input="Q",
                metric_results=[
                    MetricResult(name="M", score=0.8, passed=True, reason="OK"),
                ],
            )
        ]
    )
    # Should not raise
    er.summary()


# =============================================================================
# LLMJudge async evaluate test
# =============================================================================


@pytest.mark.asyncio
async def test_judge_evaluate_with_mock_provider():
    """Should call provider and parse response."""

    mock_provider = MagicMock()
    mock_result = MagicMock()
    mock_result.text = '{"score": 0.85, "reason": "Well written"}'
    mock_provider.generate = AsyncMock(return_value=mock_result)

    judge = LLMJudge(provider=mock_provider)
    result = await judge.evaluate(
        prompt_template="Rate: {text}",
        variables={"text": "Hello world"},
    )

    assert result["score"] == 0.85
    assert result["reason"] == "Well written"
    mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_judge_evaluate_format_template():
    """Should properly format the prompt template."""

    mock_provider = MagicMock()
    mock_result = MagicMock()
    mock_result.text = '{"score": 0.7, "reason": "OK"}'
    mock_provider.generate = AsyncMock(return_value=mock_result)

    judge = LLMJudge(provider=mock_provider)
    await judge.evaluate(
        prompt_template="Question: {q}\nAnswer: {a}",
        variables={"q": "What is AI?", "a": "Artificial Intelligence"},
    )

    call_args = mock_provider.generate.call_args[0][0]
    assert "What is AI?" in call_args
    assert "Artificial Intelligence" in call_args


def test_judge_default_provider_creation():
    """Should auto-create OpenAI provider when none is set."""
    judge = LLMJudge()

    # openai is installed in our test env
    provider = judge.provider
    assert provider is not None
    assert "OpenAI" in type(provider).__name__


def test_judge_provider_cached():
    """Provider should be cached after first access."""
    judge = LLMJudge()
    p1 = judge.provider
    p2 = judge.provider
    assert p1 is p2


# =============================================================================
# Providers __init__ lazy import
# =============================================================================


def test_providers_lazy_import_openai():
    """Should lazy-import OpenAIProvider."""
    from llmsevals import providers

    cls = providers.OpenAIProvider
    assert cls.__name__ == "OpenAIProvider"


def test_providers_lazy_import_unknown():
    """Should raise AttributeError for unknown providers."""
    from llmsevals import providers

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = providers.NonExistentProvider


# =============================================================================
# Utils edge cases
# =============================================================================


def test_tokenizer_count_message_tokens_with_model():
    """count_message_tokens should accept a model param."""
    from llmsevals.utils.tokenizer import count_message_tokens

    msgs = [{"role": "user", "content": "Hello"}]
    count = count_message_tokens(msgs, model="gpt-4o")
    assert count > 0


def test_pricing_partial_match_reversed():
    """Partial match should work when model name is a substring."""
    from llmsevals.utils.pricing import get_model_pricing

    # "gpt-4o" is in the pricing dict. Passing a longer key should still match.
    pricing = get_model_pricing("gpt-4o-2024-11-20")
    assert pricing is not None
