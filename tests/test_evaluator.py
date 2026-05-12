import pytest
pytest.importorskip("openai")
from llmsevals import Evaluator, TestCase
from llmsevals.metrics import AnswerRelevancy, Latency

@pytest.mark.parametrize("model", ["gpt-4o-mini"])
def test_simple_evaluation(model):
    evaluator = Evaluator(model=model, metrics=[AnswerRelevancy(), Latency()], verbose=False)
    tc = TestCase(input="What is AI?")
    result = evaluator.run(test_cases=[tc])
    assert result.total_test_cases == 1
    assert result.test_case_results[0].input == "What is AI?"
    assert len(result.test_case_results[0].metric_results) == 2
