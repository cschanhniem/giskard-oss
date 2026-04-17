"""Provider-specific pre-processing for OpenTelemetry GenAI payloads.

Different instrumentation packages deviate from the GenAI semantic conventions
in small, well-known ways (content shapes, event placement, bogus sentinel
messages). A :class:`ProviderNormalizer` isolates those quirks into one file
per provider so the adapters stay spec-pure.

Register new normalizers with :func:`register_normalizer` (or simply import
and extend the ``_NORMALIZERS`` dict) using the canonical provider name as
the key — the value of ``gen_ai.provider.name`` (semconv ≥1.40) or
``gen_ai.system`` (semconv ≤1.36): ``"openai"``, ``"anthropic"``,
``"gcp.gemini"``, ``"aws.bedrock"``, ``"azure.ai.openai"``,
``"azure.ai.inference"``.
"""

from .anthropic import AnthropicNormalizer
from .base import IdentityNormalizer, ProviderNormalizer

_NORMALIZERS: dict[str, ProviderNormalizer] = {
    "anthropic": AnthropicNormalizer(),
}

_IDENTITY = IdentityNormalizer()


def register_normalizer(provider: str, normalizer: ProviderNormalizer) -> None:
    """Register (or override) the normalizer for a provider name."""
    _NORMALIZERS[provider] = normalizer


def get_normalizer(provider: str | None) -> ProviderNormalizer:
    """Return the normalizer for ``provider``, or a no-op fallback."""
    if provider is None:
        return _IDENTITY
    return _NORMALIZERS.get(provider, _IDENTITY)


__all__ = [
    "AnthropicNormalizer",
    "IdentityNormalizer",
    "ProviderNormalizer",
    "get_normalizer",
    "register_normalizer",
]
