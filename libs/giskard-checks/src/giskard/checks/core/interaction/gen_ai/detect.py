"""Family and provider detection for OpenTelemetry GenAI payloads."""

from typing import Any, Literal

_EVENT_NAMES_V1_36 = {
    "gen_ai.system.message",
    "gen_ai.user.message",
    "gen_ai.assistant.message",
    "gen_ai.tool.message",
    "gen_ai.choice",
}
_EVENT_NAME_V1_40 = "gen_ai.client.inference.operation.details"

Family = Literal["events", "attributes"]


def detect_family(source: Any) -> Family:
    """Return the semconv family of ``source``.

    ``list[dict]`` of ``{event_name, body}`` is parsed as the v1.36 event
    stream unless it contains a single ``gen_ai.client.inference.operation.details``
    event, in which case its attributes are read. A raw attributes ``dict``
    containing any ``gen_ai.input.messages`` / ``gen_ai.output.messages`` key
    is parsed as v1.40+.
    """
    if isinstance(source, list):
        names = {item.get("event_name") for item in source if isinstance(item, dict)}
        if _EVENT_NAME_V1_40 in names and not (names & _EVENT_NAMES_V1_36):
            return "attributes"
        return "events"
    if isinstance(source, dict):
        if any(
            key.startswith("gen_ai.input.messages")
            or key.startswith("gen_ai.output.messages")
            for key in source
        ):
            return "attributes"
    raise ValueError(
        "Unrecognized OTel GenAI payload: expected a list of event dicts "
        "(v1.36 event stream) or an attributes dict (v1.40+)."
    )


def detect_provider(source: Any) -> str | None:
    """Best-effort extraction of the provider name from a payload.

    Reads ``gen_ai.provider.name`` (v1.40+) or ``gen_ai.system`` (v1.36),
    from either span attributes or event attributes. Returns ``None`` when
    the source does not carry the attribute.
    """
    if isinstance(source, dict):
        for key in ("gen_ai.provider.name", "gen_ai.system"):
            value = source.get(key)
            if isinstance(value, str):
                return value
    if isinstance(source, list):
        for item in source:
            if not isinstance(item, dict):
                continue
            attrs = item.get("attributes")
            if isinstance(attrs, dict):
                for key in ("gen_ai.provider.name", "gen_ai.system"):
                    value = attrs.get(key)
                    if isinstance(value, str):
                        return value
    return None


__all__ = ["Family", "detect_family", "detect_provider"]
