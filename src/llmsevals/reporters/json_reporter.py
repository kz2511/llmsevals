"""JSON file reporter for evaluation results."""

from __future__ import annotations

from pathlib import Path

from llmsevals.core.eval_result import EvalResult
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)


class JSONReporter:
    """Writes evaluation results to a JSON file.

    Example::

        from llmsevals.reporters import JSONReporter

        reporter = JSONReporter()
        reporter.report(eval_result, output_path="results/eval.json")
    """

    def report(self, eval_result: EvalResult, output_path: str | Path) -> None:
        """Write evaluation results to a JSON file.

        Args:
            eval_result: The EvalResult to serialize.
            output_path: Path to write the JSON file (parent dirs created if needed).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(eval_result.to_json(), encoding="utf-8")
        logger.info(f"JSON report written to {path.resolve()}")
