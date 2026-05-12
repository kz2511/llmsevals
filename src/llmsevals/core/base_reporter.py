"""BaseReporter — Abstract interface for evaluation report output."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmsevals.core.eval_result import EvalResult


class BaseReporter(ABC):
    """Abstract base class for evaluation reporters.

    All reporter implementations must implement the ``report()`` method
    to output evaluation results in their respective format.

    Example::

        class JsonReporter(BaseReporter):
            def report(self, eval_result: EvalResult) -> None:
                print(eval_result.to_json())
    """

    @abstractmethod
    def report(self, eval_result: EvalResult) -> None:
        """Output the evaluation results.

        Args:
            eval_result: The EvalResult to display/export.
        """
        ...
