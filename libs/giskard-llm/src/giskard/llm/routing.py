"""Route ``provider/model`` strings to the correct provider instance."""

from typing import Any

from .providers.base import BaseProvider
from .types import CompletionResponse, EmbeddingResponse

_provider_cache: dict[str, BaseProvider] = {}


def _parse_model_string(model: str) -> tuple[str, str]:
    """Split ``"provider/model-name"`` into ``(provider, model_name)``."""
    parts = model.split("/", maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid model string '{model}'. "
            "Expected format: 'provider/model-name' (e.g. 'openai/gpt-4o')."
        )
    return parts[0], parts[1]


def _get_provider(provider_name: str) -> BaseProvider:
    """Lazily instantiate and cache a provider by name."""
    if provider_name in _provider_cache:
        return _provider_cache[provider_name]

    provider = _create_provider(provider_name)
    _provider_cache[provider_name] = provider
    return provider


def _create_provider(name: str) -> BaseProvider:
    if name == "openai":
        from .providers.openai import OpenAIProvider

        return OpenAIProvider()

    if name == "gemini":
        from .providers.google import GoogleProvider

        return GoogleProvider()

    if name == "anthropic":
        from .providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    raise ValueError(
        f"Unknown provider '{name}'. Supported: openai, gemini, anthropic."
    )


async def route_completion(
    model: str,
    messages: list[dict[str, Any]],
    **params: Any,
) -> CompletionResponse:
    """Parse model string and dispatch to the right provider."""
    provider_name, model_name = _parse_model_string(model)
    provider = _get_provider(provider_name)
    return await provider.complete(model_name, messages, **params)


async def route_embedding(
    model: str,
    input: list[str],
    **params: Any,
) -> EmbeddingResponse:
    """Parse model string and dispatch to the right provider."""
    provider_name, model_name = _parse_model_string(model)
    provider = _get_provider(provider_name)
    return await provider.embed(model_name, input, **params)
