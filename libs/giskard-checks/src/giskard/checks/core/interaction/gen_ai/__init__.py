"""Parse OpenTelemetry GenAI telemetry into canonical interactions.

Two semconv families coexist in the wild: the v1.36 event stream and the
v1.40+ span-attributes payload (opt-in via
``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental``). This package
supports both and isolates per-provider deviations behind small normalizers.

Typical use:

.. code-block:: python

    from giskard.checks.core.interaction.gen_ai import GenAiTrace

    trace = GenAiTrace.from_otel(events, provider="anthropic")

See :class:`~.model.GenAiTrace` for the full public surface.
"""

from .adapters.attributes import SpanAttributesAdapter
from .adapters.events import EventStreamAdapter
from .detect import detect_family, detect_provider
from .model import (
    GenAiInteraction,
    GenAiTrace,
    Message,
    ModelResponse,
    Part,
    Role,
    TextPart,
    ToolCallPart,
    ToolCallResponsePart,
    ToolDefinition,
)
from .providers import (
    AnthropicNormalizer,
    IdentityNormalizer,
    ProviderNormalizer,
    get_normalizer,
    register_normalizer,
)

__all__ = [
    "AnthropicNormalizer",
    "EventStreamAdapter",
    "GenAiInteraction",
    "GenAiTrace",
    "IdentityNormalizer",
    "Message",
    "ModelResponse",
    "Part",
    "ProviderNormalizer",
    "Role",
    "SpanAttributesAdapter",
    "TextPart",
    "ToolCallPart",
    "ToolCallResponsePart",
    "ToolDefinition",
    "detect_family",
    "detect_provider",
    "get_normalizer",
    "register_normalizer",
]
