"""Evaluator — The main entry point for running evaluations."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone

from llmsevals.core.base_metric import BaseMetric, MetricResult
from llmsevals.core.base_provider import BaseProvider
from llmsevals.core.dataset import Dataset
from llmsevals.core.eval_result import EvalResult, TestCaseResult
from llmsevals.core.test_case import TestCase
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """The main entry point for evaluating LLM outputs.

    The Evaluator takes a list of metrics and optionally a provider,
    runs all metrics against all test cases, and returns aggregated results.

    Args:
        metrics: List of metric instances to evaluate.
        provider: Optional LLM provider. If provided, the evaluator will
            generate responses for test cases missing ``actual_output``.
        model: Shorthand for creating a default OpenAI provider with this model.
        verbose: If True, logs progress to console during evaluation.
        max_concurrency: Maximum number of concurrent metric evaluations.
            If None, all metrics for a test case run in parallel. Set to 1
            for sequential execution.

    Example::

        from llmsevals import Evaluator, TestCase, metrics

        # Simple usage
        evaluator = Evaluator(
            metrics=[
                metrics.AnswerRelevancy(threshold=0.7),
                metrics.Latency(max_latency_ms=5000),
                metrics.Cost(model="gpt-4o"),
            ]
        )

        test_cases = [
            TestCase(
                input="What is Python?",
                actual_output="Python is a programming language.",
            ),
            TestCase(
                input="What is 2+2?",
                actual_output="The answer is 4.",
            ),
        ]

        results = evaluator.run(test_cases=test_cases)
        results.summary()  # Pretty-printed console report

        # With auto-generation
        evaluator = Evaluator(
            model="gpt-4o-mini",
            metrics=[metrics.AnswerRelevancy()],
        )
        results = evaluator.run(test_cases=[
            TestCase(input="What is the capital of Japan?"),
        ])
    """

    def __init__(
        self,
        metrics: list[BaseMetric] | None = None,
        provider: BaseProvider | None = None,
        model: str | None = None,
        verbose: bool = True,
        max_concurrency: int | None = None,
        run_id: str | None = None,
        batch_size: int = 10,
    ) -> None:
        self.metrics: list[BaseMetric] = metrics or []
        self.provider = provider
        self.verbose = verbose
        self.max_concurrency = max_concurrency
        self.run_id: str = run_id or str(uuid.uuid4())[:8]
        self.batch_size = batch_size

        # Create provider from model shorthand
        if model and not provider:
            try:
                from llmsevals.providers.openai_provider import OpenAIProvider

                self.provider = OpenAIProvider(model=model)
            except ImportError:
                logger.warning(
                    f"Could not create OpenAI provider for model '{model}'. "
                    "Install with: pip install llmsevals[openai]"
                )

    def run(
        self,
        test_cases: list[TestCase] | None = None,
        dataset: Dataset | str | None = None,
        metrics: list[BaseMetric] | None = None,
        dry_run: bool = False,
    ) -> EvalResult:
        """Run evaluation synchronously.

        This is a convenience wrapper around the async ``evaluate()`` method.

        Args:
            test_cases: List of TestCase instances to evaluate.
            dataset: A Dataset instance or path to a dataset file.
            metrics: Override metrics for this run (uses instance metrics if None).
            dry_run: If True, estimate cost without running evaluations.

        Returns:
            An EvalResult with all results aggregated.
        """
        try:
            asyncio.get_running_loop()
            # Running loop detected (Jupyter, FastAPI, etc.) — try nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise RuntimeError(
                    "evaluator.run() detected a running event loop (e.g., Jupyter Notebook).\n\n"
                    "Options:\n"
                    "  1. pip install nest_asyncio  (then retry)\n"
                    "  2. Use the async API directly: await evaluator.evaluate(...)\n"
                ) from None
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.evaluate(
                    test_cases=test_cases, dataset=dataset,
                    metrics=metrics, dry_run=dry_run,
                )
            )
        except RuntimeError as _e:
            _msg = str(_e).lower()
            if "no running event loop" in _msg or "no current event loop" in _msg:
                return asyncio.run(
                    self.evaluate(
                        test_cases=test_cases, dataset=dataset,
                        metrics=metrics, dry_run=dry_run,
                    )
                )
            raise

    async def evaluate(
        self,
        test_cases: list[TestCase] | None = None,
        dataset: Dataset | str | None = None,
        metrics: list[BaseMetric] | None = None,
        dry_run: bool = False,
    ) -> EvalResult:
        """Run evaluation asynchronously.

        Args:
            test_cases: List of TestCase instances to evaluate.
            dataset: A Dataset instance or path to a dataset file.
            metrics: Override metrics for this run.
            dry_run: If True, estimate cost without running evaluations.

        Returns:
            An EvalResult with all results aggregated.
        """
        # Resolve test cases
        resolved_cases = self._resolve_test_cases(test_cases, dataset)
        if not resolved_cases:
            raise ValueError("No test cases provided. Pass test_cases or dataset.")

        # Resolve metrics
        active_metrics = metrics or self.metrics
        if not active_metrics:
            raise ValueError(
                "No metrics configured. Pass metrics to Evaluator() or run()."
            )

        # Dry-run: estimate cost without executing
        if dry_run:
            return self._estimate_cost(resolved_cases, active_metrics)

        start_time = time.perf_counter()
        metric_names = [m.name for m in active_metrics]

        from llmsevals.utils.logger import RunIdFilter
        _run_filter = RunIdFilter(self.run_id)
        logger.addFilter(_run_filter)

        try:
            return await self._run_evaluation(
                resolved_cases, active_metrics, metric_names, start_time
            )
        finally:
            logger.removeFilter(_run_filter)
            # SCALE-05: always release provider HTTP connections when done
            if self.provider is not None:
                try:
                    await self.provider.close()
                except Exception:
                    pass

    async def _run_evaluation(
        self,
        resolved_cases: list[TestCase],
        active_metrics: list[BaseMetric],
        metric_names: list[str],
        start_time: float,
    ) -> EvalResult:
        """Internal: execute the evaluation loop after setup."""
        if self.verbose:
            logger.info(
                f"Starting evaluation: {len(resolved_cases)} test cases × "
                f"{len(active_metrics)} metrics ({', '.join(metric_names)})"
            )

        # Save originals before generation (needed to interleave failure stubs)
        original_cases = list(resolved_cases)

        # Generate outputs if provider is configured and output is missing
        generation_failures: dict[int, str] = {}
        if self.provider:
            resolved_cases, generation_failures = await self._generate_outputs(resolved_cases)

        # Run evaluation in parallel batches (SCALE-01).
        # Each batch runs up to self.batch_size test cases concurrently;
        # metrics within each test case are also run in parallel.
        async def _evaluate_one(idx: int, tc: TestCase) -> TestCaseResult:
            if self.verbose:
                logger.info(f"Evaluating test case {idx + 1}/{len(resolved_cases)}")
            metric_results = await self._evaluate_metrics(tc, active_metrics, idx)
            return TestCaseResult(
                test_case_index=idx,
                input=tc.input,
                actual_output=tc.actual_output,
                metric_results=metric_results,
            )

        evaluated_results: list[TestCaseResult] = []
        for batch_start in range(0, len(resolved_cases), self.batch_size):
            batch = resolved_cases[batch_start : batch_start + self.batch_size]
            batch_results = await asyncio.gather(
                *[_evaluate_one(batch_start + j, tc) for j, tc in enumerate(batch)]
            )
            evaluated_results.extend(batch_results)

        # Merge evaluated results with generation failure stubs in original order
        result_iter = iter(evaluated_results)
        test_case_results: list[TestCaseResult] = []
        for original_idx, orig_tc in enumerate(original_cases):
            if original_idx in generation_failures:
                test_case_results.append(
                    TestCaseResult(
                        test_case_index=original_idx,
                        input=orig_tc.input,
                        actual_output=None,
                        metric_results=[],
                        generation_error=generation_failures[original_idx],
                    )
                )
            else:
                tcr = next(result_iter)
                test_case_results.append(
                    TestCaseResult(
                        test_case_index=original_idx,
                        input=tcr.input,
                        actual_output=tcr.actual_output,
                        metric_results=tcr.metric_results,
                    )
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        eval_result = EvalResult(
            test_case_results=test_case_results,
            metadata={
                "run_id": self.run_id,
                "total_time_ms": round(elapsed_ms, 2),
                "metrics": metric_names,
                "provider": str(self.provider) if self.provider else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        if self.verbose:
            logger.info(
                f"Evaluation complete in {elapsed_ms:.0f}ms — "
                f"{eval_result.passed_count}/{eval_result.total_test_cases} passed "
                f"({eval_result.pass_rate:.0%})"
            )

        return eval_result

    async def _evaluate_metrics(
        self,
        test_case: TestCase,
        active_metrics: list[BaseMetric],
        test_case_idx: int,
    ) -> list[MetricResult]:
        """Evaluate all metrics for a single test case in parallel.

        Uses asyncio.gather for concurrent execution with optional
        concurrency limiting via semaphore.

        Args:
            test_case: The test case to evaluate.
            active_metrics: List of metrics to run.
            test_case_idx: Index of the test case (for error messages).

        Returns:
            List of MetricResult, one per metric.
        """
        semaphore = (
            asyncio.Semaphore(self.max_concurrency)
            if self.max_concurrency
            else None
        )

        async def _run_metric(metric: BaseMetric) -> MetricResult:
            try:
                if semaphore:
                    async with semaphore:
                        return await metric.evaluate(test_case)
                return await metric.evaluate(test_case)
            except (AttributeError, TypeError, NameError) as e:
                # Programming bug inside the metric — re-raise so it surfaces
                # clearly instead of being silently swallowed as score=0.0.
                logger.error(
                    f"BUG in metric '{metric.name}': {type(e).__name__}: {e}"
                )
                raise
            except Exception as e:
                # Runtime / API error — record as failed metric, keep going.
                logger.error(
                    f"Metric '{metric.name}' failed on test case {test_case_idx}: "
                    f"{type(e).__name__}: {e}"
                )
                return MetricResult(
                    name=metric.name,
                    score=0.0,
                    passed=False,
                    reason=f"Evaluation error ({type(e).__name__}): {e}",
                )

        results = await asyncio.gather(
            *[_run_metric(m) for m in active_metrics]
        )
        return list(results)

    def _estimate_cost(
        self,
        test_cases: list[TestCase],
        active_metrics: list[BaseMetric],
    ) -> EvalResult:
        """Estimate evaluation cost without running any metrics.

        Args:
            test_cases: List of test cases.
            active_metrics: List of metrics that would run.

        Returns:
            An EvalResult with cost estimation metadata only.
        """
        from llmsevals.utils.pricing import estimate_cost
        from llmsevals.utils.tokenizer import count_tokens

        total_input_tokens = 0
        total_output_tokens = 0

        for tc in test_cases:
            total_input_tokens += count_tokens(tc.input)
            if tc.actual_output:
                total_output_tokens += count_tokens(tc.actual_output)

        # Estimate judge API calls: each LLM-based metric = 1 call per test case
        # Judge prompt ~300 tokens + input/output tokens per call
        judge_prompt_overhead = 300
        llm_metric_count = sum(
            1 for m in active_metrics if hasattr(m, '_judge')
        )
        judge_calls = len(test_cases) * llm_metric_count
        judge_input_tokens = judge_calls * (
            judge_prompt_overhead + (total_input_tokens + total_output_tokens)
            // max(len(test_cases), 1)
        )
        judge_output_tokens = judge_calls * 50  # ~50 tokens per JSON response

        # Use env-configured judge model for cost estimation, fallback to gpt-4o-mini
        import os as _os
        _judge_model = (
            _os.environ.get("LLMSEVALS_JUDGE_MODEL")
            or _os.environ.get("LLMSEVALS_OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        judge_cost = estimate_cost(
            _judge_model, judge_input_tokens, judge_output_tokens
        ) or 0.0

        metric_names = [m.name for m in active_metrics]
        logger.info(
            f"Dry-run estimate: {len(test_cases)} test cases × "
            f"{len(active_metrics)} metrics = {judge_calls} judge API calls"
        )
        logger.info(
            f"Estimated judge cost: ${judge_cost:.6f} "
            f"(~{judge_input_tokens} input + ~{judge_output_tokens} output tokens)"
        )

        return EvalResult(
            test_case_results=[],
            metadata={
                "dry_run": True,
                "test_case_count": len(test_cases),
                "metrics": metric_names,
                "llm_metrics": llm_metric_count,
                "estimated_judge_calls": judge_calls,
                "estimated_judge_input_tokens": judge_input_tokens,
                "estimated_judge_output_tokens": judge_output_tokens,
                "estimated_judge_cost_usd": round(judge_cost, 8),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _generate_outputs(
        self, test_cases: list[TestCase]
    ) -> tuple[list[TestCase], dict[int, str]]:
        """Generate LLM outputs for test cases missing actual_output.

        Args:
            test_cases: List of test cases, some possibly missing actual_output.

        Returns:
            Tuple of (successful_cases, {original_index: error_message}).
            Failed cases are NOT included in successful_cases.
        """
        if not self.provider:
            return test_cases, {}

        updated: list[TestCase] = []
        failures: dict[int, str] = {}

        for original_idx, tc in enumerate(test_cases):
            if not tc.has_actual_output():
                if self.verbose:
                    preview = tc.input[:60] + ("..." if len(tc.input) > 60 else "")
                    logger.info(f"Generating output for: {preview}")

                try:
                    result = await self.provider.generate(tc.input)
                    tc = tc.model_copy(
                        update={
                            "actual_output": result.text,
                            "latency_ms": result.latency_ms,
                            "token_count": result.token_count,
                            "metadata": {
                                **tc.metadata,
                                "model": result.model,
                            },
                        }
                    )
                    updated.append(tc)
                except Exception as e:
                    err_preview = tc.input[:40] + ("..." if len(tc.input) > 40 else "")
                    logger.error(f"Generation failed for '{err_preview}': {e}")
                    failures[original_idx] = str(e)
            else:
                updated.append(tc)

        return updated, failures

    def _resolve_test_cases(
        self,
        test_cases: list[TestCase] | None,
        dataset: Dataset | str | None,
    ) -> list[TestCase]:
        """Resolve test cases from direct list or dataset."""
        if test_cases:
            return test_cases

        if dataset is None:
            return []

        if isinstance(dataset, str):
            # Load from file path
            path = dataset.lower()
            if path.endswith(".jsonl"):
                return Dataset.from_jsonl(dataset).test_cases
            elif path.endswith(".csv"):
                return Dataset.from_csv(dataset).test_cases
            else:
                return Dataset.from_json(dataset).test_cases

        if isinstance(dataset, Dataset):
            return dataset.test_cases

        raise ValueError(f"Invalid dataset type: {type(dataset)}")

    def add_metric(self, metric: BaseMetric) -> Evaluator:
        """Add a metric to the evaluator (fluent API).

        Args:
            metric: A BaseMetric instance.

        Returns:
            self, for chaining.
        """
        self.metrics.append(metric)
        return self

    def __repr__(self) -> str:
        metric_names = [m.name for m in self.metrics]
        return (
            f"Evaluator(metrics={metric_names}, "
            f"provider={self.provider})"
        )
