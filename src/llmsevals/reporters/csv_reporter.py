"""CSV file reporter for evaluation results."""

from __future__ import annotations

import csv
from pathlib import Path

from llmsevals.core.eval_result import EvalResult
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)


class CSVReporter:
    """Writes per-test-case evaluation results as a CSV file.

    Each row is one test case. Columns include per-metric scores and pass/fail status.

    Example::

        from llmsevals.reporters import CSVReporter

        reporter = CSVReporter()
        reporter.report(eval_result, output_path="results/eval.csv")
    """

    def report(self, eval_result: EvalResult, output_path: str | Path) -> None:
        """Write evaluation results to a CSV file.

        Args:
            eval_result: The EvalResult to serialize.
            output_path: Path to write the CSV file (parent dirs created if needed).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not eval_result.test_case_results:
            logger.warning("No test case results to write to CSV.")
            path.write_text("", encoding="utf-8")
            return

        # Collect all metric names in first-seen order across all results
        metric_names: list[str] = []
        seen: set[str] = set()
        for tcr in eval_result.test_case_results:
            for mr in tcr.metric_results:
                if mr.name not in seen:
                    metric_names.append(mr.name)
                    seen.add(mr.name)

        base_headers = ["index", "input", "actual_output", "passed", "score", "generation_error"]
        metric_headers: list[str] = []
        for name in metric_names:
            metric_headers.append(f"{name}_score")
            metric_headers.append(f"{name}_passed")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=base_headers + metric_headers, extrasaction="ignore"
            )
            writer.writeheader()

            for tcr in eval_result.test_case_results:
                row: dict[str, object] = {
                    "index": tcr.test_case_index,
                    "input": tcr.input,
                    "actual_output": tcr.actual_output or "",
                    "passed": tcr.passed,
                    "score": round(tcr.score, 4),
                    "generation_error": tcr.generation_error or "",
                }
                for mr in tcr.metric_results:
                    row[f"{mr.name}_score"] = round(mr.score, 4)
                    row[f"{mr.name}_passed"] = mr.passed
                writer.writerow(row)

        logger.info(f"CSV report written to {path.resolve()}")
