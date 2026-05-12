"""Tests for the Dataset loader."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from llmsevals.core.dataset import Dataset
from llmsevals.core.test_case import TestCase


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    # Use workspace-local temp dir
    tmp = os.path.join(os.path.dirname(__file__), ".tmp_test_data")
    os.makedirs(tmp, exist_ok=True)
    yield tmp
    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def test_dataset_manual_creation():
    """Should create a dataset from a list of TestCases."""
    ds = Dataset([
        TestCase(input="Q1", actual_output="A1"),
        TestCase(input="Q2", actual_output="A2"),
    ])
    assert len(ds) == 2
    assert ds[0].input == "Q1"
    assert ds[1].input == "Q2"


def test_dataset_iteration():
    """Should support iteration."""
    ds = Dataset([
        TestCase(input="Q1"),
        TestCase(input="Q2"),
        TestCase(input="Q3"),
    ])
    inputs = [tc.input for tc in ds]
    assert inputs == ["Q1", "Q2", "Q3"]


def test_dataset_add():
    """Should support adding test cases."""
    ds = Dataset()
    assert len(ds) == 0

    ds.add(TestCase(input="Q1"))
    assert len(ds) == 1

    ds.add(TestCase(input="Q2"))
    assert len(ds) == 2


def test_dataset_from_list():
    """Should create from a list of dicts."""
    ds = Dataset.from_list([
        {"input": "Q1", "actual_output": "A1"},
        {"input": "Q2", "actual_output": "A2"},
    ])
    assert len(ds) == 2
    assert ds[0].actual_output == "A1"


def test_dataset_from_json(tmp_dir):
    """Should load from a JSON file."""
    data = [
        {"input": "Q1", "actual_output": "A1"},
        {"input": "Q2", "actual_output": "A2", "context": ["ctx1"]},
    ]
    path = os.path.join(tmp_dir, "test.json")
    with open(path, "w") as f:
        json.dump(data, f)

    ds = Dataset.from_json(path)
    assert len(ds) == 2
    assert ds[0].input == "Q1"
    assert ds[1].context == ["ctx1"]


def test_dataset_from_json_with_wrapper(tmp_dir):
    """Should handle JSON with 'test_cases' wrapper key."""
    data = {
        "test_cases": [
            {"input": "Q1", "actual_output": "A1"},
        ]
    }
    path = os.path.join(tmp_dir, "wrapped.json")
    with open(path, "w") as f:
        json.dump(data, f)

    ds = Dataset.from_json(path)
    assert len(ds) == 1


def test_dataset_from_jsonl(tmp_dir):
    """Should load from a JSONL file."""
    path = os.path.join(tmp_dir, "test.jsonl")
    with open(path, "w") as f:
        f.write('{"input": "Q1", "actual_output": "A1"}\n')
        f.write('{"input": "Q2", "actual_output": "A2"}\n')
        f.write("\n")  # Empty line should be skipped
        f.write('{"input": "Q3"}\n')

    ds = Dataset.from_jsonl(path)
    assert len(ds) == 3
    assert ds[2].actual_output is None


def test_dataset_from_csv(tmp_dir):
    """Should load from a CSV file."""
    path = os.path.join(tmp_dir, "test.csv")
    with open(path, "w") as f:
        f.write("input,actual_output,expected_output\n")
        f.write("Q1,A1,E1\n")
        f.write("Q2,A2,E2\n")

    ds = Dataset.from_csv(path)
    assert len(ds) == 2
    assert ds[0].input == "Q1"
    assert ds[0].expected_output == "E1"


def test_dataset_from_csv_with_context(tmp_dir):
    """CSV context should be parsed from pipe-delimited format."""
    path = os.path.join(tmp_dir, "ctx.csv")
    with open(path, "w") as f:
        f.write("input,actual_output,context\n")
        f.write("Q1,A1,ctx1|ctx2|ctx3\n")

    ds = Dataset.from_csv(path)
    assert ds[0].context == ["ctx1", "ctx2", "ctx3"]


def test_dataset_sample():
    """sample() should return a smaller dataset."""
    ds = Dataset([TestCase(input=f"Q{i}") for i in range(20)])

    sampled = ds.sample(5, seed=42)
    assert len(sampled) == 5

    # Same seed should give same result
    sampled2 = ds.sample(5, seed=42)
    assert [tc.input for tc in sampled] == [tc.input for tc in sampled2]


def test_dataset_sample_larger_than_size():
    """sample(n) where n > len should return all."""
    ds = Dataset([TestCase(input="Q1"), TestCase(input="Q2")])
    sampled = ds.sample(10)
    assert len(sampled) == 2


def test_dataset_repr():
    """__repr__ should show count."""
    ds = Dataset([TestCase(input="Q1")])
    assert "n=1" in repr(ds)


def test_dataset_test_cases_property():
    """test_cases property should return the list."""
    cases = [TestCase(input="Q1"), TestCase(input="Q2")]
    ds = Dataset(cases)
    assert ds.test_cases is cases


def test_dataset_from_json_invalid(tmp_dir):
    """Should raise on invalid JSON structure."""
    path = os.path.join(tmp_dir, "bad.json")
    with open(path, "w") as f:
        f.write('"just a string"')

    with pytest.raises(ValueError, match="Expected a list"):
        Dataset.from_json(path)


def test_dataset_from_jsonl_invalid(tmp_dir):
    """Should raise on invalid JSONL line."""
    path = os.path.join(tmp_dir, "bad.jsonl")
    with open(path, "w") as f:
        f.write('{"input": "Q1"}\n')
        f.write("not valid json\n")

    with pytest.raises(ValueError, match="Invalid JSON on line 2"):
        Dataset.from_jsonl(path)
