"""OpenTelemetry GenAI instrumentation helpers (optional extras).

Imports are lazy so ``import giskard.llm`` does not require OTEL or provider SDKs.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.instrumentation.instrumentor import (
        BaseInstrumentor,  # pyright: ignore[reportMissingImports]
    )

# provider kind -> (instrumentor module, class name, giskard-llm extra to install)
_INSTRUMENTOR_REGISTRY: dict[str, tuple[str, str, str]] = {
    "openai": (
        "opentelemetry.instrumentation.openai_v2",
        "OpenAIInstrumentor",
        "openai-otel",
    ),
    # azure/ and azure_ai/ use the OpenAI SDK -> same instrumentor, installed via azure-otel alias.
    "azure": (
        "opentelemetry.instrumentation.openai_v2",
        "OpenAIInstrumentor",
        "azure-otel",
    ),
    "azure_ai": (
        "opentelemetry.instrumentation.openai_v2",
        "OpenAIInstrumentor",
        "azure-otel",
    ),
    "google": (
        "opentelemetry.instrumentation.google_genai",
        "GoogleGenAiSdkInstrumentor",
        "google-otel",
    ),
    "anthropic": (
        "opentelemetry.instrumentation.anthropic",
        "AnthropicInstrumentor",
        "anthropic-otel",
    ),
}

# Per-provider constructor kwargs; empty dict when absent.
_INSTRUMENTOR_KWARGS: dict[str, dict[str, Any]] = {
    "anthropic": {"use_legacy_attributes": False},
}


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
    if provider not in _INSTRUMENTOR_REGISTRY:
        raise ValueError(
            f"Unknown provider kind {provider!r}. "
            f"Expected one of: {', '.join(sorted(_INSTRUMENTOR_REGISTRY))}."
        )
    module_path, class_name, extra = _INSTRUMENTOR_REGISTRY[provider]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Install giskard-llm[{extra}] (OpenTelemetry instrumentation for {provider!r}). {exc}"
        ) from exc
    cls = getattr(module, class_name)
    return cls(**_INSTRUMENTOR_KWARGS.get(provider, {}))
