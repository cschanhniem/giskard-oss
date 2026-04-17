"""Anthropic-specific normalization for OpenTelemetry GenAI payloads.

The ``opentelemetry-instrumentation-anthropic`` package deviates from the
GenAI semantic conventions in four known ways; this module concentrates all
of them so the generic adapters stay spec-pure. Each deviation is matched to
the hook that repairs it.

1. *Tool definitions leak as bogus user messages.* Before every completion
   Anthropic emits ``gen_ai.user.message`` whose ``body.content`` is
   ``{"tools": [...]}`` instead of using the semconv ``gen_ai.tool.definitions``
   attribute. :meth:`AnthropicNormalizer.normalize_events` strips those
   messages out and returns the tool definitions separately so the adapter
   can attach them to the enclosing interaction.
2. *Assistant tool calls hide in ``content`` as Anthropic content blocks.*
   ``gen_ai.assistant.message.body.content`` is a list of
   ``{"type": "tool_use", id, name, input}`` blocks instead of the semconv
   ``tool_calls`` field. :meth:`AnthropicNormalizer.parse_message_parts`
   converts those blocks into :class:`ToolCallPart`.
3. *Tool responses hide in user messages as content blocks.*
   ``gen_ai.user.message.body.content`` is a list of
   ``{"type": "tool_result", tool_use_id, content}`` blocks. The role is
   reclassified to ``"tool"`` by :meth:`AnthropicNormalizer.derive_message_role`
   and the blocks become :class:`ToolCallResponsePart` entries.
4. *Choice bodies put ``tool_calls`` at the top level* (not under
   ``message``) and the nested ``message`` object has no ``role``.
   :meth:`AnthropicNormalizer.parse_choice_body` lifts them into the
   canonical semconv shape before the adapter parses the choice.
"""

from typing import Any

from ..model import Part, TextPart, ToolCallPart, ToolCallResponsePart
from .base import ExtractedDefs, _string_content_to_parts, _tool_calls_to_parts


def _is_tool_definitions_message(body: Any) -> list[dict[str, Any]] | None:
    """Return the ``tools`` list if ``body`` is Anthropic's bogus tools shim."""
    if not isinstance(body, dict):
        return None
    content = body.get("content")
    if not isinstance(content, dict):
        return None
    tools = content.get("tools")
    if isinstance(tools, list):
        return tools
    return None


def _content_blocks_to_parts(blocks: list[Any]) -> list[Part]:
    """Convert Anthropic's native content-block array into canonical parts."""
    parts: list[Part] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(TextPart(content=text))
        elif block_type == "tool_use":
            name = block.get("name")
            if not isinstance(name, str):
                continue
            parts.append(
                ToolCallPart(
                    id=str(block.get("id", "")),
                    name=name,
                    arguments=block.get("input"),
                )
            )
        elif block_type == "tool_result":
            parts.append(
                ToolCallResponsePart(
                    id=str(block.get("tool_use_id", "")),
                    result=block.get("content"),
                )
            )
    return parts


def _tool_def_to_semconv(tool: dict[str, Any]) -> dict[str, Any]:
    """Map Anthropic's native tool shape to the semconv ``tool.definitions`` one.

    Anthropic uses ``input_schema`` for the JSON schema and omits the ``type``
    field; the semconv expects ``parameters`` and ``type`` (``function``).
    """
    normalized: dict[str, Any] = {"type": "function"}
    for key, value in tool.items():
        if key == "input_schema":
            normalized["parameters"] = value
        else:
            normalized[key] = value
    return normalized


def _content_is_only_tool_results(content: Any) -> bool:
    if not isinstance(content, list) or not content:
        return False
    return all(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


class AnthropicNormalizer:
    """Normalizer for ``opentelemetry-instrumentation-anthropic``.

    See module docstring for the list of deviations handled here.
    """

    def normalize_events(self, events: list[dict[str, Any]]) -> ExtractedDefs:
        kept: list[dict[str, Any]] = []
        extracted_tools: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for event in events:
            body = event.get("body") if isinstance(event, dict) else None
            if (
                event.get("event_name") == "gen_ai.user.message"
                and (tools := _is_tool_definitions_message(body)) is not None
            ):
                for tool in tools:
                    if not isinstance(tool, dict):
                        continue
                    name = tool.get("name")
                    if isinstance(name, str) and name in seen_names:
                        continue
                    if isinstance(name, str):
                        seen_names.add(name)
                    extracted_tools.append(_tool_def_to_semconv(tool))
                continue
            kept.append(event)
        return kept, extracted_tools

    def parse_message_parts(
        self, role: str, content: Any, tool_calls: Any | None = None
    ) -> list[Part]:
        # Anthropic encodes tool calls / results inside ``content``; the
        # semconv ``tool_calls`` field is still honored when present.
        if isinstance(content, list):
            parts = _content_blocks_to_parts(content)
        else:
            parts = list(_string_content_to_parts(content, role=role))
        parts.extend(_tool_calls_to_parts(tool_calls))
        return parts

    def parse_choice_body(self, body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        top_level_tool_calls = body.pop("tool_calls", None)
        nested = body.get("message")
        if isinstance(nested, dict):
            nested = dict(nested)
            if "role" not in nested:
                nested["role"] = "assistant"
            if top_level_tool_calls and "tool_calls" not in nested:
                nested["tool_calls"] = top_level_tool_calls
            body["message"] = nested
        elif top_level_tool_calls is not None:
            body["message"] = {
                "role": "assistant",
                "tool_calls": top_level_tool_calls,
            }
        return body

    def derive_message_role(self, declared_role: str, content: Any) -> str:
        if declared_role == "user" and _content_is_only_tool_results(content):
            return "tool"
        return declared_role
