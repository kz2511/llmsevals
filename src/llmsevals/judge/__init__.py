"""LLM-as-a-Judge system for evaluating LLM outputs."""

from llmsevals.judge.judge import LLMJudge
from llmsevals.judge.prompts import EVAL_PROMPTS

__all__ = ["LLMJudge", "EVAL_PROMPTS"]
