"""CLI runner — evaluate a single query from the terminal.

Model resolution (first match wins):
  1. --model flag
  2. LLMSEVALS_OPENAI_MODEL environment variable
  3. Default: gpt-4o-mini

Usage:
    # Use explicit model
    python examples/run_evaluator.py --model gpt-4o "What is AI?"

    # Use model from environment
    export LLMSEVALS_OPENAI_MODEL=gpt-4o-mini
    python examples/run_evaluator.py "What is AI?"
"""
import argparse
import os

from llmsevals import Evaluator, TestCase
from llmsevals.metrics import AnswerRelevancy, Latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick LLM evaluation.")
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model to use (overrides LLMSEVALS_OPENAI_MODEL env var)",
    )
    parser.add_argument("query", help="User query to evaluate")
    args = parser.parse_args()

    # Resolution: flag > env var > default
    model = args.model or os.environ.get("LLMSEVALS_OPENAI_MODEL", "gpt-4o-mini")

    evaluator = Evaluator(
        model=model,
        metrics=[AnswerRelevancy(), Latency()],
        verbose=True,
    )
    result = evaluator.run(test_cases=[TestCase(input=args.query)])
    result.summary()


if __name__ == "__main__":
    main()
