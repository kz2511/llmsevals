"""Dataset — Load test cases from files."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llmsevals.core.test_case import TestCase
from llmsevals.utils.logger import get_logger

logger = get_logger(__name__)


class Dataset:
    """Load and manage evaluation test cases from files.

    Supports JSON, JSONL (JSON Lines), and CSV formats.

    Args:
        test_cases: A list of TestCase instances.

    Example::

        # From a JSON file
        dataset = Dataset.from_json("tests.json")

        # From a JSONL file
        dataset = Dataset.from_jsonl("tests.jsonl")

        # From a CSV file
        dataset = Dataset.from_csv("tests.csv")

        # Manual creation
        dataset = Dataset([
            TestCase(input="Q1", actual_output="A1"),
            TestCase(input="Q2", actual_output="A2"),
        ])

        for test_case in dataset:
            print(test_case.input)
    """

    def __init__(self, test_cases: list[TestCase] | None = None) -> None:
        self._test_cases: list[TestCase] = test_cases or []

    @classmethod
    def from_json(cls, path: str | Path) -> Dataset:
        """Load test cases from a JSON file.

        The JSON file should contain a list of objects, where each object
        has keys matching TestCase fields (input, actual_output, etc.).

        Args:
            path: Path to the JSON file.

        Returns:
            A Dataset instance.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "test_cases" in data:
            data = data["test_cases"]

        if not isinstance(data, list):
            raise ValueError(f"Expected a list of test cases, got {type(data).__name__}")

        test_cases = [TestCase(**item) for item in data]
        return cls(test_cases)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Dataset:
        """Load test cases from a JSONL (JSON Lines) file.

        Each line should be a valid JSON object with TestCase fields.

        Args:
            path: Path to the JSONL file.

        Returns:
            A Dataset instance.
        """
        path = Path(path)
        test_cases: list[TestCase] = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {path}: {e}"
                    ) from e
                try:
                    test_cases.append(TestCase(**data))
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid TestCase fields on line {line_num} in {path}: {e}"
                    ) from e
        return cls(test_cases)

    @classmethod
    def stream_jsonl(cls, path: str | Path) -> Iterator[TestCase]:
        """Stream test cases from a JSONL file one at a time.

        Unlike ``from_jsonl()``, this is memory-efficient for large files:
        it yields one ``TestCase`` at a time without loading the entire
        file into memory. Use this when datasets exceed available RAM.

        Args:
            path: Path to the JSONL file.

        Yields:
            TestCase instances, one per non-empty line.

        Example::

            for test_case in Dataset.stream_jsonl("large_dataset.jsonl"):
                result = evaluator.run([test_case])
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {path}: {e}"
                    ) from e
                try:
                    yield TestCase(**data)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid TestCase fields on line {line_num} in {path}: {e}"
                    ) from e

    @classmethod
    def from_csv(cls, path: str | Path) -> Dataset:
        """Load test cases from a CSV file.

        The CSV should have headers matching TestCase fields.
        The 'context' column supports two formats:
          - JSON list (recommended): ["chunk one", "chunk two"]
          - Pipe-delimited (legacy): chunk one|chunk two
            WARNING: Pipe format breaks if context contains '|' (URLs, tables, code).

        Args:
            path: Path to the CSV file.

        Returns:
            A Dataset instance.
        """
        path = Path(path)
        test_cases: list[TestCase] = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle context as pipe-delimited
                processed: dict[str, Any] = {}
                for key, value in row.items():
                    if key == "context" and value:
                        stripped = value.strip()
                        if stripped.startswith("["):
                            try:
                                processed[key] = json.loads(stripped)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Context column looks like JSON but failed to parse. "
                                    f"Falling back to pipe-delimiter split. "
                                    f"Value preview: {stripped[:80]!r}"
                                )
                                processed[key] = [v.strip() for v in value.split("|")]
                        else:
                            processed[key] = [v.strip() for v in value.split("|")]
                    elif value:
                        processed[key] = value
                test_cases.append(TestCase(**processed))
        return cls(test_cases)

    @classmethod
    def from_list(cls, data: list[dict[str, Any]]) -> Dataset:
        """Create a dataset from a list of dictionaries.

        Args:
            data: List of dicts with TestCase fields.

        Returns:
            A Dataset instance.
        """
        return cls([TestCase(**item) for item in data])

    def add(self, test_case: TestCase) -> None:
        """Add a test case to the dataset."""
        self._test_cases.append(test_case)

    def sample(self, n: int, seed: int | None = None) -> Dataset:
        """Return a random sample of test cases.

        Args:
            n: Number of test cases to sample.
            seed: Random seed for reproducibility.

        Returns:
            A new Dataset with the sampled test cases.
        """
        import random

        rng = random.Random(seed)
        sampled = rng.sample(self._test_cases, min(n, len(self._test_cases)))
        return Dataset(sampled)

    @property
    def test_cases(self) -> list[TestCase]:
        """Return the list of test cases."""
        return self._test_cases

    def __len__(self) -> int:
        return len(self._test_cases)

    def __iter__(self) -> Iterator[TestCase]:
        return iter(self._test_cases)

    def __getitem__(self, index: int) -> TestCase:
        return self._test_cases[index]

    def __repr__(self) -> str:
        return f"Dataset(n={len(self._test_cases)})"
