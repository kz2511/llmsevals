"""Structured logging for llmsevals.

Log level:  controlled by LLMSEVALS_LOG_LEVEL env var (default: INFO).
Log format: controlled by LLMSEVALS_LOG_FORMAT env var (text or json).
"""

from __future__ import annotations

import logging
import os
import re
import sys

# ---------------------------------------------------------------------------
# Sensitive-data redaction
# ---------------------------------------------------------------------------
# Patterns are applied to every log message before it is emitted so that
# API keys or auth tokens that surface in exception messages never reach
# log files or aggregators.
_REDACT_RULES: list[tuple[re.Pattern[str], str]] = [
    # OpenAI / Anthropic / generic "sk-..." API keys
    (re.compile(r"\bsk-[A-Za-z0-9\-_]{16,}\b"), "[API_KEY_REDACTED]"),
    # Bearer tokens in Authorization headers
    (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]{16,}", re.IGNORECASE), "Bearer [TOKEN_REDACTED]"),
    # Explicit api_key=... or api-key:... patterns
    (re.compile(r"api[_-]?key[\"']?\s*[=:]\s*[\"']?[A-Za-z0-9\-._]{16,}", re.IGNORECASE), "api_key=[REDACTED]"),
    # Authorization header values
    (re.compile(r"(Authorization[\"']?\s*[=:]\s*[\"']?)\S{16,}", re.IGNORECASE), r"\1[REDACTED]"),
]


def _redact(message: str) -> str:
    """Replace known sensitive patterns in a log message string."""
    for pattern, replacement in _REDACT_RULES:
        message = pattern.sub(replacement, message)
    return message


class _RedactionFilter(logging.Filter):
    """Logging filter that redacts API keys and tokens from every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _redact(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: _redact(str(v)) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(_redact(str(a)) for a in record.args)
        return True

_HANDLER_NAME = "llmsevals_stderr_handler"
_LEVEL_STR = os.environ.get("LLMSEVALS_LOG_LEVEL", "INFO").upper()
_LEVEL = getattr(logging, _LEVEL_STR, logging.INFO)
_FMT = os.environ.get("LLMSEVALS_LOG_FORMAT", "text").lower()


def _make_formatter() -> logging.Formatter:
    """Return the appropriate formatter based on LLMSEVALS_LOG_FORMAT."""
    if _FMT == "json":
        import json as _json

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                entry: dict = {
                    "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if hasattr(record, "run_id"):
                    entry["run_id"] = record.run_id  # type: ignore[attr-defined]
                return _json.dumps(entry)

        return _JsonFormatter()

    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str = "llmsevals") -> logging.Logger:
    """Get a configured logger instance.

    Level is controlled by ``LLMSEVALS_LOG_LEVEL`` env var (default: INFO).
    Format is controlled by ``LLMSEVALS_LOG_FORMAT`` env var (``text`` or ``json``).

    Args:
        name: Logger name (usually module ``__name__``).

    Returns:
        A configured logging.Logger.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers across repeated calls and multiprocessing forks
    if not any(getattr(h, "name", None) == _HANDLER_NAME for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.name = _HANDLER_NAME
        handler.setFormatter(_make_formatter())
        handler.addFilter(_RedactionFilter())
        logger.addHandler(handler)

    logger.setLevel(_LEVEL)
    return logger


class RunIdFilter(logging.Filter):
    """Injects a ``run_id`` field into every log record for end-to-end traceability.

    Usage::

        filter = RunIdFilter("run-abc123")
        logger.addFilter(filter)
        # ... evaluation work ...
        logger.removeFilter(filter)
    """

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id  # type: ignore[attr-defined]
        return True
