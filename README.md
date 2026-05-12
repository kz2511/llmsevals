<p align="center">
  <h1 align="center">llmsevals</h1>
  <p align="center">A lightweight, provider-agnostic Python package for evaluating LLMs across quality, safety, cost, and latency.</p>
  <p align="center">
    <a href="https://pypi.org/project/llmsevals"><img src="https://img.shields.io/pypi/v/llmsevals?color=blue" alt="PyPI"></a>
    <a href="https://pypi.org/project/llmsevals"><img src="https://img.shields.io/pypi/pyversions/llmsevals" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  </p>
</p>

---

## Installation

```bash
# Core — Latency and Cost metrics work with no API key
pip install llmsevals

# With OpenAI — enables all LLM-as-a-Judge quality metrics
pip install "llmsevals[openai]"

# With Jupyter Notebook support
pip install "llmsevals[notebook]"

# Everything
pip install "llmsevals[all]"
```

---

## Quick Start

```python
from llmsevals import Evaluator, TestCase, metrics

evaluator = Evaluator(
    metrics=[
        metrics.Latency(threshold=0.5, max_latency_ms=5000),
        metrics.Cost(model="gpt-4o-mini", max_cost_usd=0.01),
    ]
)

results = evaluator.run(test_cases=[
    TestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        latency_ms=450.0,
        token_count={"prompt_tokens": 12, "completion_tokens": 18},
    ),
    TestCase(
        input="Explain quantum computing.",
        actual_output="Quantum computing uses qubits instead of bits.",
        latency_ms=1200.0,
        token_count={"prompt_tokens": 15, "completion_tokens": 85},
    ),
])

results.summary()
```

---

## Console Output

Running `results.summary()` prints a rich formatted report directly in your terminal:

```
╭──────────────────────────────────╮
│ 📊 llmsevals — Evaluation Report │
╰──────────────────────────────────╯
╭────────────────────────────────── Summary ───────────────────────────────────╮
│   Total Test Cases    4                                                      │
│   Passed              3                                                      │
│   Failed              1                                                      │
│   Pass Rate           75.0%                                                  │
│   Average Score       0.8323                                                 │
│   Total Time          1ms                                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
                  Metrics Overview
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Metric  ┃ Avg Score ┃ Pass Rate ┃  Min   ┃  Max   ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Latency │  0.6675   │   75.0%   │ 0.1600 │ 0.9100 │
├─────────┼───────────┼───────────┼────────┼────────┤
│ Cost    │  0.9971   │  100.0%   │ 0.9947 │ 0.9987 │
└─────────┴───────────┴───────────┴────────┴────────┘
                          Detailed Results
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #    ┃ Input                  ┃  Status  ┃  Score   ┃ Metric Details         ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1    │ What is the capital of │ ✅ PASS  │  0.9544  │ ✓ Latency: 0.91 | ✓    │
│      │ France?                │          │          │ Cost: 1.00             │
├──────┼────────────────────────┼──────────┼──────────┼────────────────────────┤
│ 2    │ Explain quantum        │ ✅ PASS  │  0.8774  │ ✓ Latency: 0.76 | ✓    │
│      │ computing.             │          │          │ Cost: 0.99             │
├──────┼────────────────────────┼──────────┼──────────┼────────────────────────┤
│ 3    │ Write a haiku about    │ ✅ PASS  │  0.9193  │ ✓ Latency: 0.84 | ✓    │
│      │ AI.                    │          │          │ Cost: 1.00             │
├──────┼────────────────────────┼──────────┼──────────┼────────────────────────┤
│ 4    │ What are benefits of   │ ❌ FAIL  │  0.5783  │ ✗ Latency: 0.16 | ✓    │
│      │ exercise?              │          │          │ Cost: 1.00             │
└──────┴────────────────────────┴──────────┴──────────┴────────────────────────┘
╭─────────────────────╮
│ ❌ 1 TEST(S) FAILED │
╰─────────────────────╯
```

---

## Evaluator

`Evaluator` is the main entry point. It runs all metrics against all inputs and returns an `EvalResult`.

```python
from llmsevals import Evaluator

evaluator = Evaluator(
    metrics=[...],             # List of metric instances
    provider=None,             # Optional: LLM provider (for auto-generation)
    model="gpt-4o-mini",       # Shorthand to create an OpenAIProvider automatically
    verbose=True,              # Log progress to console during evaluation
    max_concurrency=None,      # Max parallel metric evaluations per test case (None = unlimited)
    run_id=None,               # Custom run ID for tracing; auto-generated if None
)
```

### `run()` — Synchronous

```python
results = evaluator.run(
    test_cases=[...],    # List of TestCase instances
    dataset=dataset,     # Or pass a Dataset / path to file
    metrics=[...],       # Override metrics for this run only
    dry_run=False,       # If True, estimates cost without calling any APIs
)
```

### `evaluate()` — Asynchronous

```python
results = await evaluator.evaluate(
    test_cases=[...],
    dataset=dataset,
    metrics=[...],
    dry_run=False,
)
```

### Auto-Generation

If a `TestCase` has no `actual_output`, the evaluator calls the provider to generate one:

```python
evaluator = Evaluator(
    model="gpt-4o-mini",
    metrics=[metrics.AnswerRelevancy()],
)

results = evaluator.run(test_cases=[
    TestCase(input="What is the capital of Japan?"),   # No actual_output — will be generated
    TestCase(input="Explain photosynthesis briefly."),
])
```

### Dry Run — Estimate Cost

```python
estimate = evaluator.run(test_cases=test_cases, dry_run=True)

print(estimate.metadata["estimated_judge_cost_usd"])
print(estimate.metadata["estimated_judge_calls"])
```

---

## Metrics

### Latency — no API key required

Scores response time linearly. Lower latency = higher score. Score = `1.0 - (latency_ms / max_latency_ms)`, clamped to `[0, 1]`.

```python
from llmsevals.metrics import Latency

Latency(
    max_latency_ms=5000,   # Latency at which score hits 0.0 (default: 10000)
    threshold=0.5,         # Minimum score to pass (default: 0.5)
)
```

**Requires:** `test_case.latency_ms`

| latency_ms | max_latency_ms | Score |
|---:|---:|:---:|
| 0 | 5000 | 1.00 |
| 500 | 5000 | 0.90 |
| 2500 | 5000 | 0.50 |
| 4200 | 5000 | 0.16 |
| 5000+ | 5000 | 0.00 |

---

### Cost — no API key required

Estimates USD cost from token counts using built-in pricing data. Score = `1.0 - (cost_usd / max_cost_usd)`, clamped to `[0, 1]`.

```python
from llmsevals.metrics import Cost

Cost(
    model="gpt-4o-mini",   # Model to look up pricing for
    max_cost_usd=0.01,     # Cost at which score hits 0.0 (default: 0.10)
    threshold=0.5,
)
```

**Requires:** `test_case.token_count` with `prompt_tokens` and `completion_tokens`.  
Falls back to counting tokens from text if `token_count` is not set.

**Built-in pricing for:**

| Provider | Models |
|----------|--------|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`, `o1`, `o1-mini`, `o3-mini` |
| Anthropic | `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`, `claude-3.5-sonnet`, `claude-3.5-haiku`, `claude-opus-4`, `claude-sonnet-4` |
| Google | `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-2.0-flash`, `gemini-2.5-pro`, `gemini-2.5-flash` |
| Meta | `llama-3.1-70b`, `llama-3.1-405b`, `llama-3.3-70b` |

Add or override pricing at runtime:

```python
from llmsevals.utils import pricing

pricing.update_pricing("my-custom-model", input_per_1k=0.001, output_per_1k=0.002)
```

---

### AnswerRelevancy — requires `OPENAI_API_KEY`

How directly and completely does the answer address the question?

```python
from llmsevals.metrics import AnswerRelevancy

AnswerRelevancy(threshold=0.7)
```

**Requires:** `input`, `actual_output`

---

### Faithfulness — requires `OPENAI_API_KEY`

Is every claim in the answer grounded in the provided context?

```python
from llmsevals.metrics import Faithfulness

Faithfulness(threshold=0.8)
```

**Requires:** `actual_output`, `context` (list of strings)

---

### Hallucination — requires `OPENAI_API_KEY`

Does the answer contain claims that contradict or are not supported by the context?  
High score (→ 1.0) means the model is **not** hallucinating.

```python
from llmsevals.metrics import Hallucination

Hallucination(threshold=0.7)
```

**Requires:** `actual_output`, `context`

---

### Toxicity — requires `OPENAI_API_KEY`

Is the content safe and free from harmful language?

```python
from llmsevals.metrics import Toxicity

Toxicity(threshold=0.8)
```

**Requires:** `actual_output`

---

### Bias — requires `OPENAI_API_KEY`

Does the answer treat all groups fairly without discriminatory language?

```python
from llmsevals.metrics import Bias

Bias(threshold=0.7)
```

**Requires:** `actual_output`

---

### Coherence — requires `OPENAI_API_KEY`

Is the answer grammatically correct, logically structured, and easy to follow?

```python
from llmsevals.metrics import Coherence

Coherence(threshold=0.6)
```

**Requires:** `actual_output`

---

### AnswerCorrectness — requires `OPENAI_API_KEY`

How close is the answer to the expected ground-truth answer?

```python
from llmsevals.metrics import AnswerCorrectness

AnswerCorrectness(threshold=0.7)
```

**Requires:** `actual_output`, `expected_output`

---

### All Metrics Summary

| Metric | API Key | Required Fields | Default Threshold |
|--------|:-------:|-----------------|:-----------------:|
| `Latency` | No | `latency_ms` | 0.5 |
| `Cost` | No | `token_count` | 0.5 |
| `AnswerRelevancy` | Yes | `input`, `actual_output` | 0.5 |
| `Faithfulness` | Yes | `actual_output`, `context` | 0.7 |
| `Hallucination` | Yes | `actual_output`, `context` | 0.7 |
| `Toxicity` | Yes | `actual_output` | 0.8 |
| `Bias` | Yes | `actual_output` | 0.7 |
| `Coherence` | Yes | `actual_output` | 0.5 |
| `AnswerCorrectness` | Yes | `actual_output`, `expected_output` | 0.7 |

---

## Providers

### OpenAIProvider

```python
from llmsevals.providers import OpenAIProvider

provider = OpenAIProvider(
    model="gpt-4o",           # Any OpenAI model. Reads LLMSEVALS_OPENAI_MODEL env var if not set.
    api_key="sk-...",         # Reads OPENAI_API_KEY env var if not set.
    base_url=None,            # Custom base URL for Azure OpenAI or compatible APIs.
    temperature=0.0,          # Default: 0.0
    max_tokens=1024,          # Default: 1024
)

result = await provider.generate("What is 2 + 2?")
print(result.text)            # "4"
print(result.latency_ms)      # e.g. 312.4
print(result.prompt_tokens)   # e.g. 12
print(result.completion_tokens)  # e.g. 5
print(result.model)           # "gpt-4o-mini" (from API response)
```

Use a provider directly with the evaluator:

```python
evaluator = Evaluator(provider=provider, metrics=[...])

# or shorthand (creates OpenAIProvider internally):
evaluator = Evaluator(model="gpt-4o", metrics=[...])
```

### Custom Provider

Extend `BaseProvider` to support any LLM:

```python
import time
from llmsevals.core import BaseProvider, GenerationResult

class MyProvider(BaseProvider):
    def __init__(self):
        super().__init__(model="my-model")

    async def generate(self, prompt: str) -> GenerationResult:
        start = time.perf_counter()
        text = call_my_llm(prompt)          # your API call here
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            model=self.model,
        )

    async def generate_with_messages(self, messages: list[dict]) -> GenerationResult:
        prompt = "\n".join(m["content"] for m in messages)
        return await self.generate(prompt)
```

---

## Dataset Loading

```python
from llmsevals import Dataset

# From JSON list file
dataset = Dataset.from_json("tests.json")

# From JSONL (one JSON object per line)
dataset = Dataset.from_jsonl("tests.jsonl")

# From CSV
dataset = Dataset.from_csv("tests.csv")

# From a list of dicts
dataset = Dataset.from_list([
    {"input": "Q1", "actual_output": "A1"},
    {"input": "Q2", "actual_output": "A2"},
])

# Sample randomly (for quick checks)
sampled = dataset.sample(n=100, seed=42)

print(len(dataset))   # number of items
```

**JSONL format:**
```jsonl
{"input": "What is AI?", "actual_output": "AI is...", "latency_ms": 500, "token_count": {"prompt_tokens": 8, "completion_tokens": 20}}
{"input": "Define ML.", "actual_output": "ML is...", "latency_ms": 700, "token_count": {"prompt_tokens": 6, "completion_tokens": 18}}
```

**CSV format (context as JSON list):**
```csv
input,actual_output,expected_output,latency_ms,context
"What is AI?","AI is...","Artificial Intelligence",500,"[""AI stands for Artificial Intelligence""]"
```

Pass a dataset directly to `run()`:

```python
results = evaluator.run(dataset=dataset)

# Or pass a file path directly
results = evaluator.run(dataset="tests.jsonl")
```

---

## Reporters

### Console

Called automatically by `results.summary()`. Uses Rich with plain-text fallback.

```python
results.summary()
```

### JSON Reporter

```python
from llmsevals.reporters import JSONReporter

JSONReporter().report(results, "results.json")
```

**Output structure:**
```json
{
  "total_test_cases": 4,
  "passed": 3,
  "failed": 1,
  "generation_errors": 0,
  "pass_rate": 0.75,
  "average_score": 0.8323,
  "metric_summary": {
    "Latency": {"avg_score": 0.6675, "pass_rate": 0.75, "min_score": 0.16, "max_score": 0.91, "count": 4},
    "Cost":    {"avg_score": 0.9971, "pass_rate": 1.0,  "min_score": 0.9947, "max_score": 0.9987, "count": 4}
  },
  "test_case_results": [
    {
      "index": 0,
      "input": "What is the capital of France?",
      "actual_output": "The capital of France is Paris.",
      "generation_error": null,
      "passed": true,
      "score": 0.9544,
      "metrics": [
        {
          "name": "Latency",
          "score": 0.91,
          "passed": true,
          "reason": "Response time: 450ms (max allowed: 5000ms)",
          "metadata": {"latency_ms": 450.0, "max_latency_ms": 5000}
        },
        {
          "name": "Cost",
          "score": 0.9987,
          "passed": true,
          "reason": "Estimated cost: $0.000013 (max budget: $0.0100)",
          "metadata": {"cost_usd": 1.26e-05, "model": "gpt-4o-mini", "prompt_tokens": 12, "completion_tokens": 18, "total_tokens": 30, "max_cost_usd": 0.01}
        }
      ]
    }
  ],
  "metadata": {
    "run_id": "ca34b33b",
    "total_time_ms": 1.06,
    "metrics": ["Latency", "Cost"],
    "provider": null,
    "timestamp": "2026-05-11T08:47:53.651525+00:00"
  }
}
```

### CSV Reporter

```python
from llmsevals.reporters import CSVReporter

CSVReporter().report(results, "results.csv")
```

One row per test case. Metric scores appear as separate columns:

```csv
index,input,actual_output,passed,score,generation_error,Latency,Cost
0,What is the capital of France?,The capital of France is Paris.,True,0.9544,,0.91,0.9987
1,Explain quantum computing.,Quantum computing uses qubits...,True,0.8774,,0.76,0.9947
2,Write a haiku about AI.,Silicon dreams flow...,True,0.9193,,0.84,0.9985
3,What are benefits of exercise?,Exercise improves...,False,0.5783,,0.16,0.9965
```

### HTML Reporter

```python
from llmsevals.reporters import HTMLReporter

HTMLReporter().report(results, "results.html")
# Open results.html in any browser — no internet connection required
```

### Use all reporters together

```python
from llmsevals.reporters import JSONReporter, CSVReporter, HTMLReporter

results = evaluator.run(test_cases=test_cases)
results.summary()                                 # Console
JSONReporter().report(results, "results.json")
CSVReporter().report(results, "results.csv")
HTMLReporter().report(results, "results.html")
```

---

## Accessing Results Programmatically

```python
results = evaluator.run(test_cases=test_cases)

# Aggregates
results.total_test_cases       # int
results.passed_count           # int
results.failed_count           # int
results.generation_error_count # int — cases where provider failed before any metric ran
results.pass_rate              # float 0.0–1.0
results.average_score          # float 0.0–1.0

# Per-metric breakdown
for metric_name, stats in results.metric_summary().items():
    print(metric_name, stats["avg_score"], stats["pass_rate"], stats["min_score"], stats["max_score"])

# Per-test-case
for tc in results.test_case_results:
    print(tc.input, tc.passed, tc.score, tc.generation_error)
    for m in tc.metric_results:
        print(f"  {m.name}: {m.score:.4f} ({'pass' if m.passed else 'fail'}) — {m.reason}")

# Serialise
data = results.to_dict()   # dict
json_str = results.to_json()  # JSON string (raw_response excluded)
```

---

## Custom Metrics

Extend `BaseMetric`. The framework handles pass/fail, aggregation, and all reporters automatically.

```python
from llmsevals.core import BaseMetric, MetricResult

class ResponseLength(BaseMetric):
    """Penalises responses that are too short or too long."""

    def __init__(self, min_words: int = 10, max_words: int = 200, threshold: float = 0.5):
        super().__init__(threshold=threshold)
        self.min_words = min_words
        self.max_words = max_words

    @property
    def name(self) -> str:
        return "ResponseLength"

    @property
    def required_fields(self) -> list[str]:
        return ["actual_output"]

    async def evaluate(self, test_case) -> MetricResult:
        if not test_case.actual_output:
            return self._make_result(score=0.0, reason="No output")

        word_count = len(test_case.actual_output.split())

        if word_count < self.min_words:
            score = word_count / self.min_words
            reason = f"Too short: {word_count} words (min {self.min_words})"
        elif word_count > self.max_words:
            score = self.max_words / word_count
            reason = f"Too long: {word_count} words (max {self.max_words})"
        else:
            score = 1.0
            reason = f"Good length: {word_count} words"

        return self._make_result(score=score, reason=reason)
```

---

## Environment Variables

All variables are optional.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (generation and judging) |
| `LLMSEVALS_OPENAI_MODEL` | `gpt-4o-mini` | Default model for `OpenAIProvider` when `model` is not passed |
| `LLMSEVALS_JUDGE_MODEL` | value of `LLMSEVALS_OPENAI_MODEL` | Model used by the LLM judge. Falls back to `LLMSEVALS_OPENAI_MODEL` then `gpt-4o-mini` |
| `LLMSEVALS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LLMSEVALS_LOG_FORMAT` | `text` | `text` or `json` (for log aggregators) |
| `LLMSEVALS_SKIP_PRICING_WARNING` | — | Set to any value to suppress stale-pricing warnings |

**Model resolution order for `OpenAIProvider`:**
1. `model=` argument passed to constructor
2. `LLMSEVALS_OPENAI_MODEL` env var
3. Hard fallback: `gpt-4o-mini`

**Model resolution order for LLM judge:**
1. `LLMSEVALS_JUDGE_MODEL` env var
2. `LLMSEVALS_OPENAI_MODEL` env var
3. Hard fallback: `gpt-4o-mini`

---

## CI/CD Example

```python
# tests/test_quality_gate.py
from llmsevals import Evaluator, TestCase, metrics

def test_quality_gate():
    evaluator = Evaluator(
        metrics=[
            metrics.Latency(threshold=0.5, max_latency_ms=5000),
            metrics.Cost(model="gpt-4o-mini", max_cost_usd=0.01),
        ]
    )

    results = evaluator.run(test_cases=[
        TestCase(
            input="Summarise AI in one sentence.",
            actual_output="AI enables machines to simulate human intelligence.",
            latency_ms=800.0,
            token_count={"prompt_tokens": 10, "completion_tokens": 12},
        ),
    ])

    assert results.pass_rate >= 0.8, f"Quality gate failed: {results.pass_rate:.1%}"
```

```bash
pytest tests/test_quality_gate.py
```

---

## Roadmap

### v1.0.0 — Current (OpenAI)
- 9 metrics: `Latency`, `Cost`, `AnswerRelevancy`, `Faithfulness`, `Hallucination`, `Toxicity`, `Bias`, `Coherence`, `AnswerCorrectness`
- OpenAI provider with async support and token/latency tracking
- LLM-as-a-Judge with prompt injection protection and exponential backoff retry
- 4 reporters: Console, JSON, CSV, HTML
- Dataset loading from JSON, JSONL, CSV
- Env-var model selection: `LLMSEVALS_OPENAI_MODEL`, `LLMSEVALS_JUDGE_MODEL`
- Dry-run cost estimation
- Jupyter-compatible via `nest_asyncio`

### v2.0.0 — In Progress (Multi-Provider)
- Anthropic provider — Claude 3.5 / Claude 4 family
- Google provider — Gemini 2.0 / 2.5 family
- Ollama provider — local models
- Model comparison mode — same test cases across multiple providers side by side

---

## License

MIT — see [LICENSE](LICENSE).
