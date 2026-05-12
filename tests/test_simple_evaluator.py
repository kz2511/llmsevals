import pytest
from llmsevals import Evaluator, TestCase
from llmsevals.metrics import Latency

@pytest.mark.parametrize("model", ["gpt-4o-mini"])
def test_simple_evaluation(model):
    # Use a metric that does not require external API calls.
    evaluator = Evaluator(provider=None, metrics=[Latency(max_latency_ms=10000)], verbose=False)
    tc = TestCase(
        input="What is AI?",
        actual_output="AI stands for Artificial Intelligence, a field of computer science.",
        latency_ms=500,
    )
    result = evaluator.run(test_cases=[tc])
    assert result.total_test_cases == 1
    assert result.test_case_results[0].input == "What is AI?"
    # Latency metric should produce a result.
    assert len(result.test_case_results[0].metric_results) == 1
    # The score should be >0 (since latency is less than max).
    assert result.test_case_results[0].metric_results[0].score > 0
