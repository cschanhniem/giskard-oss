"""Route ``provider/model`` strings to the correct provider instance."""

import importlib
import os
from typing import Any

from .providers.base import BaseProvider
from .types import ChatMessage, CompletionResponse, EmbeddingResponse

_PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "openai": ("giskard.llm.providers.openai", "OpenAIProvider"),
    "google": ("giskard.llm.providers.google", "GoogleProvider"),
    "gemini": ("giskard.llm.providers.google", "GoogleProvider"),
    "anthropic": ("giskard.llm.providers.anthropic", "AnthropicProvider"),
    "azure": ("giskard.llm.providers.azure_openai", "AzureOpenAIProvider"),
    "azure_ai": ("giskard.llm.providers.azure_ai", "AzureAIProvider"),
}


def _resolve_value(value: Any) -> Any:
    """Resolve ``os.environ/VAR_NAME`` strings to env var values."""
    if isinstance(value, str) and value.startswith("os.environ/"):
        var_name = value[len("os.environ/") :]
        return os.environ.get(var_name)
    return value


def _parse_model_string(model: str) -> tuple[str, str]:
    """Split ``"provider/model-name"`` into ``(provider, model_name)``.

    Bare model names (no ``/``) default to ``"openai"``.
    """
    model = model.strip()
    if not model:
        raise ValueError(
            "Invalid model string ''. "
            "Expected format: 'provider/model-name' (e.g. 'openai/gpt-4o')."
        )
    if "/" not in model:
        return "openai", model
    parts = model.split("/", maxsplit=1)
    provider, model_name = parts[0].strip(), parts[1].strip()
    if not provider or not model_name:
        raise ValueError(
            f"Invalid model string '{model}'. "
            "Expected format: 'provider/model-name' (e.g. 'openai/gpt-4o')."
        )
    return provider, model_name


def _create_provider(provider_type: str, **kwargs: Any) -> BaseProvider:
    """Instantiate a provider by type name using the registry."""
    if provider_type not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider_type}'. "
            f"Supported: {', '.join(sorted(_PROVIDER_REGISTRY))}."
        )
    module_path, class_name = _PROVIDER_REGISTRY[provider_type]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


class LLMClient:
    """Entry point for configuring and calling LLM providers.

    Stores config from ``configure()`` calls. Provider instances are
    created lazily on first use and cached on this client instance.
    """

    def __init__(self) -> None:
        self._configs: dict[str, dict[str, Any]] = {}
        self._providers: dict[str, BaseProvider] = {}

    def configure(self, name: str, provider: str | None = None, **kwargs: Any) -> None:
        """Register a named provider configuration.

        Args:
            name: Alias for this provider (used as model string prefix).
            provider: Provider type from the registry. Defaults to *name*.
            **kwargs: Connection config (api_key, base_url, ...) and
                behavior config (merge_system, ...). Values may use
                ``os.environ/VAR_NAME`` syntax for deferred env var resolution.
        """
        name = name.strip()
        provider_type = (provider or name).strip()
        self._configs[name] = {"provider": provider_type, **kwargs}
        self._providers.pop(name, None)

    def configure_from_dict(self, config: dict[str, dict[str, Any]]) -> None:
        """Bulk-register providers from a dict (e.g. loaded from YAML)."""
        for name, kwargs in config.items():
            self.configure(name, **kwargs)

    def _get_provider(self, name: str) -> BaseProvider:
        if name in self._providers:
            return self._providers[name]

        if name in self._configs:
            cfg = dict(self._configs[name])
            provider_type = cfg.pop("provider")
            resolved = {k: _resolve_value(v) for k, v in cfg.items()}
            provider = _create_provider(provider_type, **resolved)
            self._providers[name] = provider
            return provider

        if name in _PROVIDER_REGISTRY:
            provider = _create_provider(name)
            self._providers[name] = provider
            return provider

        raise ValueError(
            f"Provider '{name}' is not configured and not in the registry. "
            f"Call client.configure('{name}', ...) first."
        )

    async def acompletion(
        self,
        model: str,
        messages: list[ChatMessage],
        **params: Any,
    ) -> CompletionResponse:
        """Parse model string and dispatch to the right provider."""
        alias, model_name = _parse_model_string(model)
        provider = self._get_provider(alias)
        return await provider.complete(model_name, messages, **params)

    async def aembedding(
        self,
        model: str,
        input: list[str],
        **params: Any,
    ) -> EmbeddingResponse:
        """Parse model string and dispatch to the right provider."""
        alias, model_name = _parse_model_string(model)
        provider = self._get_provider(alias)
        return await provider.embed(model_name, input, **params)


_default_client = LLMClient()


async def route_completion(
    model: str,
    messages: list[ChatMessage],
    **params: Any,
) -> CompletionResponse:
    """Module-level convenience wrapper around the default client."""
    return await _default_client.acompletion(model, messages, **params)


async def route_embedding(
    model: str,
    input: list[str],
    **params: Any,
) -> EmbeddingResponse:
    """Module-level convenience wrapper around the default client."""
    return await _default_client.aembedding(model, input, **params)
