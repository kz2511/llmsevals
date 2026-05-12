"""CLI entry point for quick evaluation runs.
Usage:
    python -m llmsevals --model gpt-4o-mini --query "What is AI?"
"""
import argparse
from llmsevals import Evaluator, TestCase
from llmsevals.metrics import AnswerRelevancy, Latency

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick LLM evaluation.")
    parser.add_argument("--model", type=str, required=True, help="Model shorthand for OpenAI provider")
    parser.add_argument("--query", type=str, required=True, help="User query to evaluate")
    args = parser.parse_args()

    evaluator = Evaluator(
        model=args.model,
        metrics=[AnswerRelevancy(), Latency()],
    )
    test_cases = [TestCase(input=args.query)]
    result = evaluator.run(test_cases=test_cases)
    result.summary()

if __name__ == "__main__":
    main()
