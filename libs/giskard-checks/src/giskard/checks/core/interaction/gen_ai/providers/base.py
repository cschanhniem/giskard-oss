"""Provider normalizer protocol and no-op default."""

from typing import Any, Protocol, runtime_checkable

from ..model import Part, TextPart, ToolCallPart, ToolCallResponsePart

ExtractedDefs = tuple[list[dict[str, Any]], list[dict[str, Any]]]
"""``(events_without_bogus, extracted_tool_definitions)``."""


@runtime_checkable
class ProviderNormalizer(Protocol):
    """Hooks for mapping provider-native OTel payloads into canonical parts.

    Each hook has a sensible default in :class:`IdentityNormalizer`; providers
    override only what they need to. Hooks are intentionally small and pure
    (``dict -> dict`` / ``Any -> list[Part]``) so they compose, test, and
    evolve independently of the adapters.
    """

    def normalize_events(self, events: list[dict[str, Any]]) -> ExtractedDefs: ...

    def parse_message_parts(
        self, role: str, content: Any, tool_calls: Any | None = None
    ) -> list[Part]: ...

    def parse_choice_body(self, body: dict[str, Any]) -> dict[str, Any]: ...

    def derive_message_role(self, declared_role: str, content: Any) -> str: ...


class IdentityNormalizer:
    """Default no-op normalizer used for spec-compliant providers.

    Implements :class:`ProviderNormalizer`. Applies the minimum mapping that
    the base GenAI semconv promises: a ``str`` ``content`` becomes a single
    :class:`TextPart`, and ``tool_calls`` on an assistant message are folded
    into :class:`ToolCallPart` entries.
    """

    def normalize_events(self, events: list[dict[str, Any]]) -> ExtractedDefs:
        return list(events), []

    def parse_message_parts(
        self, role: str, content: Any, tool_calls: Any | None = None
    ) -> list[Part]:
        parts: list[Part] = []
        parts.extend(_string_content_to_parts(content, role=role))
        parts.extend(_tool_calls_to_parts(tool_calls))
        return parts

    def parse_choice_body(self, body: dict[str, Any]) -> dict[str, Any]:
        # Some instrumentations omit ``role`` inside the nested message; assume
        # assistant for a ``gen_ai.choice`` since choices are always model output.
        body = dict(body)
        nested = body.get("message")
        if isinstance(nested, dict) and "role" not in nested:
            body["message"] = {"role": "assistant", **nested}
        return body

    def derive_message_role(self, declared_role: str, content: Any) -> str:
        return declared_role


def _string_content_to_parts(content: Any, *, role: str) -> list[Part]:
    """Map a string/None ``content`` into canonical parts.

    ``role == "tool"`` is treated specially: a string becomes a
    :class:`ToolCallResponsePart` with an empty id (callers SHOULD fill it
    from the event body's ``id``).
    """
    if content is None or content == "":
        return []
    if isinstance(content, str):
        if role == "tool":
            return [ToolCallResponsePart(id="", result=content)]
        return [TextPart(content=content)]
    # Non-string content at this layer is provider-specific — the normalizer
    # for that provider is responsible for turning it into parts before this
    # helper is reached.
    return []


def _tool_calls_to_parts(tool_calls: Any | None) -> list[ToolCallPart]:
    """Map a semconv-shaped ``tool_calls`` list into :class:`ToolCallPart`."""
    if not isinstance(tool_calls, list):
        return []
    parts: list[ToolCallPart] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        parts.append(
            ToolCallPart(
                id=str(call.get("id", "")),
                name=name,
                arguments=function.get("arguments"),
            )
        )
    return parts
