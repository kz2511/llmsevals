"""Tests for the Latency metric."""

from __future__ import annotations

import pytest

from llmsevals.core.test_case import TestCase
from llmsevals.metrics.latency import Latency


@pytest.mark.asyncio
async def test_latency_fast_response():
    """Fast response should get a high score."""
    metric = Latency(max_latency_ms=5000)
    tc = TestCase(
        input="test",
        actual_output="response",
        latency_ms=500.0,
    )
    result = await metric.evaluate(tc)

    assert result.score == 0.9
    assert result.passed is True
    assert result.name == "Latency"
    assert "500ms" in result.reason


@pytest.mark.asyncio
async def test_latency_slow_response():
    """Slow response should get a low score."""
    metric = Latency(threshold=0.5, max_latency_ms=5000)
    tc = TestCase(
        input="test",
        actual_output="response",
        latency_ms=4000.0,
    )
    result = await metric.evaluate(tc)

    assert result.score == pytest.approx(0.2, abs=0.01)
    assert result.passed is False


@pytest.mark.asyncio
async def test_latency_exceeds_max():
    """Response exceeding max latency should score 0."""
    metric = Latency(max_latency_ms=3000)
    tc = TestCase(
        input="test",
        actual_output="response",
        latency_ms=5000.0,
    )
    result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert result.passed is False


@pytest.mark.asyncio
async def test_latency_zero():
    """Zero latency should score 1.0."""
    metric = Latency(max_latency_ms=5000)
    tc = TestCase(
        input="test",
        actual_output="response",
        latency_ms=0.0,
    )
    result = await metric.evaluate(tc)

    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.asyncio
async def test_latency_no_data():
    """Missing latency data should score 0 with a warning."""
    metric = Latency()
    tc = TestCase(
        input="test",
        actual_output="response",
    )
    result = await metric.evaluate(tc)

    assert result.score == 0.0
    assert result.passed is False
    assert "missing_latency" in result.metadata.get("warning", "")


@pytest.mark.asyncio
async def test_latency_metadata():
    """Metadata should contain latency details."""
    metric = Latency(max_latency_ms=10000)
    tc = TestCase(
        input="test",
        actual_output="response",
        latency_ms=2500.0,
    )
    result = await metric.evaluate(tc)

    assert result.metadata["latency_ms"] == 2500.0
    assert result.metadata["max_latency_ms"] == 10000.0


@pytest.mark.asyncio
async def test_latency_seconds_display():
    """Latency above 1s should display in seconds."""
    metric = Latency(max_latency_ms=10000)
    tc = TestCase(
        input="test",
        actual_output="response",
        latency_ms=3500.0,
    )
    result = await metric.evaluate(tc)

    assert "3.50s" in result.reason
