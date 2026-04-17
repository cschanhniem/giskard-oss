"""Span-attribute adapter for OpenTelemetry GenAI semconv ≥1.40.

Consumes a single ``gen_ai.client.inference.operation.details`` payload whose
attributes include ``gen_ai.input.messages``, ``gen_ai.output.messages``,
``gen_ai.system_instructions``, and ``gen_ai.tool.definitions`` — already in
the canonical parts-based shape the spec is converging on. Each payload maps
to exactly one :class:`GenAiInteraction`; callers that have many spans should
build one trace per span and concatenate.
"""

import json
from typing import Any

from ..model import (
    GenAiInteraction,
    Message,
    ModelResponse,
    Part,
    TextPart,
    ToolCallPart,
    ToolCallResponsePart,
    ToolDefinition,
)
from ..providers.base import IdentityNormalizer, ProviderNormalizer

_INPUT_MESSAGES_ATTR = "gen_ai.input.messages"
_OUTPUT_MESSAGES_ATTR = "gen_ai.output.messages"
_SYSTEM_INSTRUCTIONS_ATTR = "gen_ai.system_instructions"
_TOOL_DEFINITIONS_ATTR = "gen_ai.tool.definitions"


def _coerce_attribute(value: Any) -> Any:
    """Attributes are sometimes serialized as JSON strings on spans."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _part_from_dict(raw: dict[str, Any]) -> Part | None:
    part_type = raw.get("type")
    if part_type == "text":
        content = raw.get("content")
        if isinstance(content, str):
            return TextPart(content=content)
    elif part_type == "tool_call":
        name = raw.get("name")
        if isinstance(name, str):
            return ToolCallPart(
                id=str(raw.get("id", "")),
                name=name,
                arguments=raw.get("arguments"),
            )
    elif part_type == "tool_call_response":
        return ToolCallResponsePart(
            id=str(raw.get("id", "")),
            result=raw.get("result"),
        )
    return None


def _parts_from_raw(raw_parts: Any) -> list[Part]:
    if not isinstance(raw_parts, list):
        return []
    parts: list[Part] = []
    for item in raw_parts:
        if isinstance(item, dict) and (part := _part_from_dict(item)) is not None:
            parts.append(part)
    return parts


class SpanAttributesAdapter:
    """Parser for the v1.40+ attributes family (single span, structured parts)."""

    family = "attributes"

    def parse(
        self,
        attributes: dict[str, Any],
        *,
        normalizer: ProviderNormalizer | None = None,
    ) -> list[GenAiInteraction]:
        normalizer = normalizer or IdentityNormalizer()

        system_instructions = _coerce_attribute(
            attributes.get(_SYSTEM_INSTRUCTIONS_ATTR, [])
        )
        input_messages = _coerce_attribute(attributes.get(_INPUT_MESSAGES_ATTR, []))
        output_messages = _coerce_attribute(attributes.get(_OUTPUT_MESSAGES_ATTR, []))
        tool_definitions = _coerce_attribute(attributes.get(_TOOL_DEFINITIONS_ATTR, []))

        system_parts = [
            p for p in _parts_from_raw(system_instructions) if isinstance(p, TextPart)
        ]

        inputs: list[Message] = []
        if isinstance(input_messages, list):
            for idx, raw in enumerate(input_messages):
                if not isinstance(raw, dict):
                    continue
                role = raw.get("role", "user")
                parts = _parts_from_raw(raw.get("parts"))
                effective_role = normalizer.derive_message_role(role, parts)
                inputs.append(
                    Message(role=effective_role, parts=parts)  # pyright: ignore[reportArgumentType]
                )
                _ = idx  # reserved for ``finish_reason``/index propagation if needed

        outputs: list[ModelResponse] = []
        if isinstance(output_messages, list):
            for idx, raw in enumerate(output_messages):
                if not isinstance(raw, dict):
                    continue
                role = raw.get("role", "assistant")
                parts = _parts_from_raw(raw.get("parts"))
                outputs.append(
                    ModelResponse(
                        role=role,  # pyright: ignore[reportArgumentType]
                        parts=parts,
                        finish_reason=(
                            raw.get("finish_reason")
                            if isinstance(raw.get("finish_reason"), str)
                            else None
                        ),
                        index=idx,
                    )
                )

        tool_defs: list[ToolDefinition] = []
        if isinstance(tool_definitions, list):
            for raw in tool_definitions:
                if isinstance(raw, dict):
                    tool_defs.append(ToolDefinition.model_validate(raw))

        if not inputs and not outputs:
            return []
        return [
            GenAiInteraction(
                inputs=inputs,
                outputs=outputs,
                system_instructions=system_parts,
                tool_definitions=tool_defs,
            )
        ]
