"""LLM provider integrations."""

from llmsevals.core.base_provider import BaseProvider, GenerationResult

__all__ = ["BaseProvider", "GenerationResult"]

# Lazy imports for optional providers
def __getattr__(name: str):  # type: ignore[misc]
    if name == "OpenAIProvider":
        from llmsevals.providers.openai_provider import OpenAIProvider
        return OpenAIProvider
    raise AttributeError(f"module 'llmsevals.providers' has no attribute {name!r}")
