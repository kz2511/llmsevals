"""Standalone HTML reporter — no external dependencies, works offline."""

from __future__ import annotations

from pathlib import Path

from llmsevals.core.eval_result import EvalResult
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)

_GREEN = "#22c55e"
_RED = "#ef4444"
_AMBER = "#f59e0b"


def _esc(text: str) -> str:
    """HTML-escape a string to prevent XSS in the report."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


class HTMLReporter:
    """Writes a self-contained HTML evaluation report.

    No external dependencies — pure HTML + inline CSS. Works offline and in CI.

    Example::

        from llmsevals.reporters import HTMLReporter

        reporter = HTMLReporter()
        reporter.report(eval_result, output_path="results/eval.html")
    """

    def report(self, eval_result: EvalResult, output_path: str | Path) -> None:
        """Write evaluation results to a standalone HTML file.

        Args:
            eval_result: The EvalResult to serialize.
            output_path: Path to write the HTML file (parent dirs created if needed).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._build_html(eval_result), encoding="utf-8")
        logger.info(f"HTML report written to {path.resolve()}")

    def _build_html(self, r: EvalResult) -> str:
        run_id = _esc(str(r.metadata.get("run_id", "N/A")))
        timestamp = _esc(str(r.metadata.get("timestamp", "N/A")))
        provider = _esc(str(r.metadata.get("provider", "N/A")))
        metric_names: list[str] = r.metadata.get("metrics", [])

        summary_rows = "".join(
            f"<tr><td>{_esc(name)}</td>"
            f"<td>{stats['avg_score']:.3f}</td>"
            f"<td>{stats['pass_rate']:.1%}</td>"
            f"<td>{stats['min_score']:.3f}</td>"
            f"<td>{stats['max_score']:.3f}</td></tr>"
            for name, stats in r.metric_summary().items()
        )

        detail_rows = ""
        for tcr in r.test_case_results:
            if tcr.generation_error:
                status_cell = f'<td style="color:{_AMBER}">⚠ GEN ERROR</td>'
            elif tcr.passed:
                status_cell = f'<td style="color:{_GREEN}">✓ PASS</td>'
            else:
                status_cell = f'<td style="color:{_RED}">✗ FAIL</td>'

            metric_cells = "".join(
                f'<td style="color:{_GREEN if mr.passed else _RED}">{mr.score:.3f}</td>'
                for mr in tcr.metric_results
            )

            input_preview = _esc((tcr.input or "")[:100])
            output_preview = _esc((tcr.actual_output or tcr.generation_error or "")[:100])

            detail_rows += (
                f"<tr>"
                f"<td>{tcr.test_case_index}</td>"
                f"<td>{input_preview}</td>"
                f"<td>{output_preview}</td>"
                f"{status_cell}"
                f"<td>{tcr.score:.3f}</td>"
                f"{metric_cells}"
                f"</tr>"
            )

        metric_th = "".join(f"<th>{_esc(n)}</th>" for n in metric_names)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llmsevals Report — {run_id}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; color: #1f2937; background: #fff; }}
  h1 {{ color: #111827; margin-bottom: 0.25rem; }}
  h2 {{ color: #374151; margin-top: 2rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.5rem; }}
  .meta {{ color: #6b7280; font-size: 0.875rem; margin-bottom: 1.5rem; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem 1.5rem; text-align: center; min-width: 110px; }}
  .card .value {{ font-size: 2rem; font-weight: 700; }}
  .card .label {{ font-size: 0.8rem; color: #6b7280; margin-top: 0.25rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.75rem; font-size: 0.875rem; }}
  th {{ background: #f9fafb; padding: 10px 12px; text-align: left; border: 1px solid #e5e7eb; font-weight: 600; }}
  td {{ padding: 8px 12px; border: 1px solid #e5e7eb; vertical-align: top; word-break: break-word; max-width: 300px; }}
  tr:nth-child(even) td {{ background: #f9fafb; }}
</style>
</head>
<body>
<h1>llmsevals Evaluation Report</h1>
<p class="meta">
  Run ID: <strong>{run_id}</strong> &nbsp;|&nbsp;
  {timestamp} &nbsp;|&nbsp;
  Provider: {provider}
</p>

<div class="cards">
  <div class="card">
    <div class="value" style="color:{_GREEN}">{r.passed_count}</div>
    <div class="label">Passed</div>
  </div>
  <div class="card">
    <div class="value" style="color:{_RED}">{r.failed_count}</div>
    <div class="label">Failed</div>
  </div>
  <div class="card">
    <div class="value" style="color:{_AMBER}">{r.generation_error_count}</div>
    <div class="label">Gen Errors</div>
  </div>
  <div class="card">
    <div class="value">{r.pass_rate:.1%}</div>
    <div class="label">Pass Rate</div>
  </div>
  <div class="card">
    <div class="value">{r.average_score:.3f}</div>
    <div class="label">Avg Score</div>
  </div>
</div>

<h2>Metric Summary</h2>
<table>
  <tr>
    <th>Metric</th>
    <th>Avg Score</th>
    <th>Pass Rate</th>
    <th>Min</th>
    <th>Max</th>
  </tr>
  {summary_rows if summary_rows else "<tr><td colspan='5'>No metric data</td></tr>"}
</table>

<h2>Detailed Results ({r.total_test_cases} test cases)</h2>
<table>
  <tr>
    <th>#</th>
    <th>Input</th>
    <th>Output / Error</th>
    <th>Status</th>
    <th>Score</th>
    {metric_th}
  </tr>
  {detail_rows if detail_rows else "<tr><td colspan='5'>No results</td></tr>"}
</table>
</body>
</html>"""
