"""Adapters that turn an OpenTelemetry GenAI payload into canonical domain objects.

Two adapters cover the two semconv families that coexist in the wild:

- :class:`EventStreamAdapter` — semconv ≤1.36, one log per message plus
  ``gen_ai.choice`` entries.
- :class:`SpanAttributesAdapter` — semconv ≥1.40, a single
  ``gen_ai.client.inference.operation.details`` payload with structured
  ``gen_ai.input.messages`` / ``gen_ai.output.messages`` attributes.

:func:`parse_source` auto-detects the family and dispatches; callers that
know which shape they have should use the adapters (or the
:class:`~giskard.checks.core.interaction.gen_ai.model.GenAiTrace` factories)
directly.
"""

from typing import Any

from ..detect import detect_family, detect_provider
from ..model import GenAiInteraction
from ..providers import get_normalizer
from .attributes import SpanAttributesAdapter
from .events import EventStreamAdapter


def parse_source(
    source: Any,
    *,
    provider: str | None,
    raise_on_unknown_event: bool,
    drop_redundant_input_history: bool,
) -> list[GenAiInteraction]:
    """Auto-detect the semconv family and parse ``source`` into interactions."""
    family = detect_family(source)
    normalizer = get_normalizer(provider or detect_provider(source))
    if family == "events":
        adapter = EventStreamAdapter(
            raise_on_unknown_event=raise_on_unknown_event,
            drop_redundant_input_history=drop_redundant_input_history,
        )
        return adapter.parse(source, normalizer=normalizer)
    adapter_attrs = SpanAttributesAdapter()
    return adapter_attrs.parse(source, normalizer=normalizer)


__all__ = [
    "EventStreamAdapter",
    "SpanAttributesAdapter",
    "parse_source",
]
