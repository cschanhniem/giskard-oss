"""OpenTelemetry GenAI instrumentation helpers (optional extras).

Imports are lazy so ``import giskard.llm`` does not require OTEL or provider SDKs.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.instrumentation.instrumentor import BaseInstrumentor


def instrumentor_for_provider(provider: str) -> "BaseInstrumentor":
    """Return the OpenTelemetry GenAI instrumentor for a provider kind.

    Azure OpenAI and Azure AI Foundry use the OpenAI SDK; they share the same
    instrumentation as ``openai``.

    Args:
        provider: Provider kind (e.g. ``openai``, ``google``, ``anthropic``,
            ``azure``, ``azure_ai``).

    Returns:
        An instrumentor instance with ``instrument()`` / ``uninstrument()``.

    Raises:
        ImportError: If the matching ``giskard-llm`` OTEL extra is not installed.
        ValueError: If ``provider`` is not a known provider kind.
    """
    if provider in ("openai", "azure", "azure_ai"):
        try:
            from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

            return OpenAIInstrumentor()
        except ImportError as exc:
            raise ImportError(
                f"Install giskard-llm[{provider},{provider}-otel] "
                f"(OpenTelemetry OpenAI instrumentation). {exc}"
            ) from exc
    if provider == "google":
        try:
            from opentelemetry.instrumentation.google_genai import (
                GoogleGenAiSdkInstrumentor,
            )

            return GoogleGenAiSdkInstrumentor()
        except ImportError as exc:
            raise ImportError(
                "Install giskard-llm[google,google-otel] "
                f"(OpenTelemetry Google GenAI instrumentation). {exc}"
            ) from exc
    if provider == "anthropic":
        try:
            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

            return AnthropicInstrumentor(use_legacy_attributes=False)
        except ImportError as exc:
            raise ImportError(
                "Install giskard-llm[anthropic,anthropic-otel] "
                f"(OpenTelemetry Anthropic instrumentation). {exc}"
            ) from exc
    raise ValueError(
        f"Unknown provider kind {provider!r}. "
        f"Expected one of: openai, google, anthropic, azure, azure_ai."
    )
