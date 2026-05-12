"""Console reporter — Rich-formatted terminal output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmsevals.core.base_reporter import BaseReporter

if TYPE_CHECKING:
    from llmsevals.core.eval_result import EvalResult


class ConsoleReporter(BaseReporter):
    """Pretty-print evaluation results to the console using Rich.

    Produces a colorful, well-formatted table showing per-test-case
    and per-metric results, plus aggregate summary statistics.

    Example::

        from llmsevals.reporters import ConsoleReporter

        reporter = ConsoleReporter()
        reporter.report(eval_result)
    """

    def report(self, eval_result: EvalResult) -> None:
        """Print a formatted evaluation report to the console.

        Args:
            eval_result: The EvalResult to display.
        """
        try:
            import rich  # noqa: F401 — check availability only

            self._rich_report(eval_result)
        except ImportError:
            self._plain_report(eval_result)

    def _rich_report(self, eval_result: EvalResult) -> None:
        """Rich-formatted report with tables and colors."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Header
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]📊 llmsevals — Evaluation Report[/bold cyan]",
                border_style="cyan",
            )
        )

        # Summary stats
        summary_table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        summary_table.add_column("Key", style="bold")
        summary_table.add_column("Value")

        summary_table.add_row("Total Test Cases", str(eval_result.total_test_cases))
        summary_table.add_row(
            "Passed",
            f"[green]{eval_result.passed_count}[/green]",
        )
        summary_table.add_row(
            "Failed",
            f"[red]{eval_result.failed_count}[/red]" if eval_result.failed_count > 0 else "0",
        )

        pass_rate = eval_result.pass_rate
        rate_color = "green" if pass_rate >= 0.8 else "yellow" if pass_rate >= 0.5 else "red"
        summary_table.add_row(
            "Pass Rate",
            f"[{rate_color}]{pass_rate:.1%}[/{rate_color}]",
        )
        summary_table.add_row(
            "Average Score",
            f"{eval_result.average_score:.4f}",
        )

        if eval_result.metadata.get("total_time_ms"):
            time_ms = eval_result.metadata["total_time_ms"]
            if time_ms < 1000:
                time_str = f"{time_ms:.0f}ms"
            else:
                time_str = f"{time_ms / 1000:.2f}s"
            summary_table.add_row("Total Time", time_str)

        console.print(Panel(summary_table, title="Summary", border_style="blue"))

        # Per-metric summary
        metric_summary = eval_result.metric_summary()
        if metric_summary:
            metric_table = Table(
                title="Metrics Overview",
                show_lines=True,
            )
            metric_table.add_column("Metric", style="bold cyan")
            metric_table.add_column("Avg Score", justify="center")
            metric_table.add_column("Pass Rate", justify="center")
            metric_table.add_column("Min", justify="center")
            metric_table.add_column("Max", justify="center")

            for name, stats in metric_summary.items():
                avg = stats["avg_score"]
                pr = stats["pass_rate"]
                avg_color = "green" if avg >= 0.7 else "yellow" if avg >= 0.4 else "red"
                pr_color = "green" if pr >= 0.8 else "yellow" if pr >= 0.5 else "red"

                metric_table.add_row(
                    name,
                    f"[{avg_color}]{avg:.4f}[/{avg_color}]",
                    f"[{pr_color}]{pr:.1%}[/{pr_color}]",
                    f"{stats['min_score']:.4f}",
                    f"{stats['max_score']:.4f}",
                )

            console.print(metric_table)

        # Detailed results table
        if eval_result.test_case_results:
            detail_table = Table(
                title="Detailed Results",
                show_lines=True,
            )
            detail_table.add_column("#", style="dim", width=4)
            detail_table.add_column("Input", max_width=40)
            detail_table.add_column("Status", justify="center", width=8)
            detail_table.add_column("Score", justify="center", width=8)
            detail_table.add_column("Metric Details", max_width=60)

            for tcr in eval_result.test_case_results:
                status = "[green]✅ PASS[/green]" if tcr.passed else "[red]❌ FAIL[/red]"
                input_preview = tcr.input[:37] + "..." if len(tcr.input) > 37 else tcr.input

                # Build metric details
                details_parts = []
                for mr in tcr.metric_results:
                    icon = "✓" if mr.passed else "✗"
                    color = "green" if mr.passed else "red"
                    details_parts.append(
                        f"[{color}]{icon} {mr.name}: {mr.score:.2f}[/{color}]"
                    )
                details = " | ".join(details_parts)

                detail_table.add_row(
                    str(tcr.test_case_index + 1),
                    input_preview,
                    status,
                    f"{tcr.score:.4f}",
                    details,
                )

            console.print(detail_table)

        # Footer
        console.print()
        overall_status = (
            "[bold green]✅ ALL TESTS PASSED[/bold green]"
            if eval_result.failed_count == 0
            else f"[bold red]❌ {eval_result.failed_count} TEST(S) FAILED[/bold red]"
        )
        console.print(Panel.fit(overall_status, border_style="cyan"))
        console.print()

    def _plain_report(self, eval_result: EvalResult) -> None:
        """Fallback plain-text report when Rich is not available."""
        print()
        print("=" * 60)
        print("  llmsevals — Evaluation Report")
        print("=" * 60)
        print()
        print(f"  Total Test Cases: {eval_result.total_test_cases}")
        print(f"  Passed:           {eval_result.passed_count}")
        print(f"  Failed:           {eval_result.failed_count}")
        print(f"  Pass Rate:        {eval_result.pass_rate:.1%}")
        print(f"  Average Score:    {eval_result.average_score:.4f}")
        print()

        # Metric summary
        metric_summary = eval_result.metric_summary()
        if metric_summary:
            print("-" * 60)
            print(f"  {'Metric':<25} {'Avg':>8} {'Pass%':>8} {'Min':>8} {'Max':>8}")
            print("-" * 60)
            for name, stats in metric_summary.items():
                print(
                    f"  {name:<25} {stats['avg_score']:>8.4f} "
                    f"{stats['pass_rate']:>7.1%} {stats['min_score']:>8.4f} "
                    f"{stats['max_score']:>8.4f}"
                )
            print()

        # Detailed results
        for tcr in eval_result.test_case_results:
            status = "PASS" if tcr.passed else "FAIL"
            print(f"  [{status}] #{tcr.test_case_index + 1}: {tcr.input[:50]}")
            for mr in tcr.metric_results:
                icon = "✓" if mr.passed else "✗"
                print(f"         {icon} {mr.name}: {mr.score:.4f} — {mr.reason}")
            print()

        # Footer
        if eval_result.failed_count == 0:
            print("  ✅ ALL TESTS PASSED")
        else:
            print(f"  ❌ {eval_result.failed_count} TEST(S) FAILED")
        print()
