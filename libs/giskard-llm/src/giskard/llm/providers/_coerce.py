"""Pydantic boundary-coercion helpers shared across provider implementations.

Providers accept either typed Pydantic models or plain dicts at their public
surface and want to operate on normalized dicts internally. These helpers
run a ``TypeAdapter`` validation once at the boundary and re-serialize with
``model_dump()`` so downstream code (``_validate_messages``, ``_convert_*``)
keeps using ``.get(...)``-style access on OpenAI-shaped dicts.

Content parts that reduce to a single text segment are collapsed back to a
plain string so the internal dict shape matches the legacy
``{"role": "user", "content": "..."}`` wire format most providers expect.
"""

from collections.abc import Sequence
from typing import Any

from pydantic import TypeAdapter

from ..types import (
    FunctionTool,
    FunctionToolDefinition,
    Message,
)

_MSG_ADAPTER: TypeAdapter[list[Message]] = TypeAdapter(list[Message])
_COMPLETION_TOOL_ADAPTER: TypeAdapter[list[FunctionToolDefinition]] = TypeAdapter(
    list[FunctionToolDefinition]
)
_RESPONSE_TOOL_ADAPTER: TypeAdapter[list[FunctionTool]] = TypeAdapter(
    list[FunctionTool]
)


def _flatten_text_content(value: Any) -> Any:
    """Reduce ``[{"type": "text", "text": "..."}, ...]`` back to a plain string.

    Providers internally use the legacy OpenAI Chat Completions wire shape
    (content is a plain string). After ``Message.model_dump()`` content is a
    list of typed parts; flatten it back when all parts are text so the
    provider-internal dict processing keeps working unchanged.
    """
    if not isinstance(value, list):
        return value
    if not value:
        return ""
    texts: list[str] = []
    for part in value:
        if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
            texts.append(part["text"])
        else:
            # Non-text / mixed content — return the list unchanged so providers
            # that know how to handle structured content (or the SDK itself)
            # can deal with it.
            return value
    return "\n".join(texts) if texts else ""


def _flatten_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Strip internal metadata and flatten content for legacy OpenAI-style dicts."""
    if "content" in msg and msg["content"] is not None:
        msg["content"] = _flatten_text_content(msg["content"])
    # Drop the discriminator tag that InputMessage dumps — it's not part of
    # the OpenAI Chat Completions wire format and would be rejected by the SDK.
    msg.pop("type", None)
    return msg


def coerce_messages(
    messages: Sequence[Message | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate a mixed sequence of Message/dict and return legacy-shaped dicts."""
    validated = _MSG_ADAPTER.validate_python(list(messages))
    return [_flatten_message(m.model_dump()) for m in validated]


def coerce_completion_tools(
    tools: Sequence[FunctionToolDefinition | dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Validate Chat Completions tools (nested ``{type, function: {...}}``)."""
    if tools is None:
        return None
    validated = _COMPLETION_TOOL_ADAPTER.validate_python(list(tools))
    return [t.model_dump() for t in validated]


def coerce_response_tools(
    tools: Sequence[FunctionTool | dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Validate Responses API tools (flat ``{type, name, description, parameters}``)."""
    if tools is None:
        return None
    validated = _RESPONSE_TOOL_ADAPTER.validate_python(list(tools))
    return [t.model_dump() for t in validated]
