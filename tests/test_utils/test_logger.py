"""Tests for structured logging — env var config, dedup, RunIdFilter."""

from __future__ import annotations

import json as _json
import logging
import os
from unittest.mock import patch

import pytest


def test_get_logger_returns_logger():
    from llmsevals.utils.logger import get_logger
    logger = get_logger("test.basic")
    assert isinstance(logger, logging.Logger)


def test_no_duplicate_handlers():
    from llmsevals.utils.logger import get_logger
    logger = get_logger("test.dedup.unique123")
    before = len(logger.handlers)
    get_logger("test.dedup.unique123")  # second call
    assert len(logger.handlers) == before


def test_handler_has_name():
    from llmsevals.utils.logger import _HANDLER_NAME, get_logger
    logger = get_logger("test.handler.name")
    names = [getattr(h, "name", None) for h in logger.handlers]
    assert _HANDLER_NAME in names


def test_run_id_filter_injects_field():
    from llmsevals.utils.logger import RunIdFilter

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("test.run_id_filter")
    capture = _Capture()
    logger.addHandler(capture)
    logger.setLevel(logging.DEBUG)

    flt = RunIdFilter("abc-123")
    logger.addFilter(flt)
    logger.info("test message")
    logger.removeFilter(flt)

    assert records, "No log records captured"
    assert hasattr(records[-1], "run_id")
    assert records[-1].run_id == "abc-123"  # type: ignore[attr-defined]


def test_run_id_filter_allows_all_records():
    from llmsevals.utils.logger import RunIdFilter
    flt = RunIdFilter("x")
    record = logging.LogRecord("n", logging.INFO, "", 0, "msg", (), None)
    assert flt.filter(record) is True


def test_run_id_in_evaluator_metadata():
    from unittest.mock import AsyncMock, patch

    from llmsevals.core.base_metric import BaseMetric, MetricResult
    from llmsevals.core.test_case import TestCase
    from llmsevals.runners.evaluator import Evaluator

    class _FakeMetric(BaseMetric):
        @property
        def name(self) -> str:
            return "Fake"

        async def evaluate(self, test_case):
            return MetricResult(name="Fake", score=1.0, passed=True)

    evaluator = Evaluator(metrics=[_FakeMetric()], run_id="my-run-xyz")
    tc = TestCase(input="Q", actual_output="A")
    result = evaluator.run(test_cases=[tc])
    assert result.metadata.get("run_id") == "my-run-xyz"


def test_evaluator_auto_generates_run_id():
    from llmsevals.core.base_metric import BaseMetric, MetricResult
    from llmsevals.core.test_case import TestCase
    from llmsevals.runners.evaluator import Evaluator

    class _FakeMetric(BaseMetric):
        @property
        def name(self) -> str:
            return "Fake"

        async def evaluate(self, test_case):
            return MetricResult(name="Fake", score=1.0, passed=True)

    evaluator = Evaluator(metrics=[_FakeMetric()])
    assert evaluator.run_id is not None
    assert len(evaluator.run_id) > 0


# =============================================================================
# Missing coverage tests for 95%+ coverage (JSON formatter)
# =============================================================================


def test_json_formatter_format():
    """JSON formatter should output structured JSON with required fields."""
    from llmsevals.utils import logger as logger_module

    with patch.object(logger_module, "_FMT", "json"):
        formatter = logger_module._make_formatter()

        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        parsed = _json.loads(formatted)

        assert "timestamp" in parsed
        assert "level" in parsed
        assert parsed["level"] == "INFO"
        assert "logger" in parsed
        assert parsed["logger"] == "test.logger"
        assert "message" in parsed
        assert parsed["message"] == "Test message"


def test_json_formatter_with_run_id():
    """JSON formatter should include run_id when present on record."""
    from llmsevals.utils import logger as logger_module

    with patch.object(logger_module, "_FMT", "json"):
        formatter = logger_module._make_formatter()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.run_id = "test-run-123"  # type: ignore[attr-defined]

        formatted = formatter.format(record)
        parsed = _json.loads(formatted)

        assert "run_id" in parsed
        assert parsed["run_id"] == "test-run-123"


def test_text_formatter_format():
    """Text formatter should output human-readable format."""
    from llmsevals.utils import logger as logger_module

    with patch.object(logger_module, "_FMT", "text"):
        formatter = logger_module._make_formatter()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Warning message",
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        assert "WARNING" in formatted
        assert "test.logger" in formatted
        assert "Warning message" in formatted
