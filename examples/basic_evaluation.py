"""
basic_evaluation.py — llmsevals Quick Start Example

Demonstrates three tiers of usage:
  1. Local metrics (Latency & Cost) — no API key needed
  2. LLM-as-a-Judge metrics — requires OPENAI_API_KEY
  3. Full pipeline with auto-generation — requires OPENAI_API_KEY

Model selection via environment variables (no code changes needed):
  export LLMSEVALS_OPENAI_MODEL=gpt-4o       # generation model
  export LLMSEVALS_JUDGE_MODEL=gpt-4o-mini   # judge model (defaults to above)
  export OPENAI_API_KEY=sk-...               # your OpenAI API key

Usage:
    python examples/basic_evaluation.py
"""

import os

from llmsevals import Evaluator, TestCase, metrics
from llmsevals.reporters import JSONReporter


def example_1_local_metrics():
    """Latency & Cost — zero API calls, works offline."""
    print("=" * 60)
    print("  Example 1: Latency & Cost (no API key needed)")
    print("=" * 60)

    # Model for Cost is the model whose pricing table to use.
    # Read from env so users can switch without touching code.
    cost_model = os.environ.get("LLMSEVALS_OPENAI_MODEL", "gpt-4o-mini")

    evaluator = Evaluator(
        metrics=[
            metrics.Latency(threshold=0.5, max_latency_ms=5000),
            metrics.Cost(model=cost_model, max_cost_usd=0.05),
        ],
        verbose=True,
    )

    test_cases = [
        TestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            latency_ms=450.0,
            token_count={"prompt_tokens": 12, "completion_tokens": 18},
        ),
        TestCase(
            input="Explain quantum computing in simple terms.",
            actual_output=(
                "Quantum computing uses quantum bits (qubits) instead of classical bits. "
                "Unlike classical bits that are either 0 or 1, qubits can be in a "
                "superposition of both states simultaneously."
            ),
            latency_ms=1200.0,
            token_count={"prompt_tokens": 15, "completion_tokens": 85},
        ),
        TestCase(
            input="Write a haiku about programming.",
            actual_output="Code flows like water\nBugs emerge from hidden depths\nDebug, test, repeat",
            latency_ms=800.0,
            token_count={"prompt_tokens": 10, "completion_tokens": 25},
        ),
        TestCase(
            input="What are the benefits of exercise?",
            actual_output=(
                "Exercise improves cardiovascular health, strengthens muscles, "
                "boosts mental health, aids weight management, and improves sleep quality."
            ),
            latency_ms=3500.0,  # Slow — will fail latency threshold
            token_count={"prompt_tokens": 12, "completion_tokens": 65},
        ),
    ]

    results = evaluator.run(test_cases=test_cases)
    results.summary()

    # Per-metric breakdown
    print()
    for metric_name, stats in results.metric_summary().items():
        print(f"  {metric_name}: avg={stats['avg_score']:.3f}  pass_rate={stats['pass_rate']:.0%}")

    # CI/CD assertion pattern
    assert results.pass_rate >= 0.5, f"Pass rate too low: {results.pass_rate:.1%}"
    print("\nCI/CD assertion passed: pass_rate >= 50%")
    return results


def example_2_judge_metrics():
    """LLM-as-a-Judge metrics — requires OPENAI_API_KEY."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nSkipping Example 2 — set OPENAI_API_KEY to run judge metrics.\n")
        return

    print("\n" + "=" * 60)
    print("  Example 2: LLM-as-a-Judge Metrics")
    print("=" * 60)

    # LLMSEVALS_JUDGE_MODEL controls which model evaluates outputs.
    # LLMSEVALS_OPENAI_MODEL is the fallback if JUDGE_MODEL is not set.
    # Both default to gpt-4o-mini if neither is set.
    evaluator = Evaluator(
        metrics=[
            metrics.AnswerRelevancy(threshold=0.7),
            metrics.Faithfulness(threshold=0.8),
            metrics.Hallucination(threshold=0.7),
        ],
        verbose=True,
    )

    test_cases = [
        TestCase(
            input="What is the boiling point of water?",
            actual_output="Water boils at 100°C (212°F) at standard atmospheric pressure.",
            context=["Water has a boiling point of 100 degrees Celsius at 1 atm pressure."],
        ),
        TestCase(
            input="Who invented the telephone?",
            actual_output="The telephone was invented by Alexander Graham Bell in 1876.",
            context=["Alexander Graham Bell is credited with inventing the telephone in 1876."],
        ),
    ]

    results = evaluator.run(test_cases=test_cases)
    results.summary()
    return results


def example_3_json_export(results):
    """Export results to JSON."""
    print("\n" + "=" * 60)
    print("  Example 3: JSON Export")
    print("=" * 60 + "\n")

    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    JSONReporter().report(results, path)
    print(f"JSON report saved to: {path}")
    print(results.to_json()[:500], "...")
    _os.unlink(path)


if __name__ == "__main__":
    results = example_1_local_metrics()
    example_2_judge_metrics()
    example_3_json_export(results)
