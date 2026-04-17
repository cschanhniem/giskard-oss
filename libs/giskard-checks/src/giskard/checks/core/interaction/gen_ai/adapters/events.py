"""Event-stream adapter for OpenTelemetry GenAI semconv ≤1.36.

Consumes the sequence of log records emitted by instrumentations that follow
the v1.36 spec: ``gen_ai.system.message``, ``gen_ai.user.message``,
``gen_ai.assistant.message``, ``gen_ai.tool.message``, ``gen_ai.choice``.
A new interaction is flushed every time a non-choice event follows one or
more ``gen_ai.choice`` events (choice → next message boundary).

Per-provider deviations live in :class:`~..providers.base.ProviderNormalizer`
implementations; this adapter only sees canonicalized events.
"""

import logging
from typing import Any

from ..model import (
    GenAiInteraction,
    Message,
    ModelResponse,
    ToolCallResponsePart,
    ToolDefinition,
)
from ..providers.base import IdentityNormalizer, ProviderNormalizer

logger = logging.getLogger(__name__)


_ROLE_BY_EVENT: dict[str, str] = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}


def _event_name_and_body(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    event_name = event.get("event_name")
    if not isinstance(event_name, str):
        raise ValueError(f"Event name is not a string: event={event}")
    body = event.get("body")
    if body is None:
        raise ValueError(
            "Event body is missing; ensure OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT "
            f"is enabled: {event}"
        )
    if not isinstance(body, dict):
        raise ValueError(f"Event body is not a dict: {body}")
    return event_name, body


def _start_index_after_last_assistant(messages: list[Message]) -> int:
    """Index that keeps only messages after the last ``assistant`` entry."""
    last_assistant = -1
    for i, m in enumerate(messages):
        if m.role == "assistant":
            last_assistant = i
    return last_assistant + 1


class EventStreamAdapter:
    """Parser for the v1.36 event-stream family.

    Parameters
    ----------
    raise_on_unknown_event
        If ``True`` (default), unknown ``event_name`` values raise
        :class:`ValueError`; otherwise they are skipped.
    drop_redundant_input_history
        Some instrumentations repeat the whole conversation in the message
        stream before each completion. When ``True``, each flushed
        interaction keeps only messages after the last ``assistant`` turn in
        its inputs.
    """

    family = "events"

    def __init__(
        self,
        *,
        raise_on_unknown_event: bool = True,
        drop_redundant_input_history: bool = False,
    ) -> None:
        self.raise_on_unknown_event = raise_on_unknown_event
        self.drop_redundant_input_history = drop_redundant_input_history

    def parse(
        self,
        events: list[dict[str, Any]],
        *,
        normalizer: ProviderNormalizer | None = None,
    ) -> list[GenAiInteraction]:
        normalizer = normalizer or IdentityNormalizer()
        events, extracted_tools = normalizer.normalize_events(events)
        shared_tool_definitions = [
            ToolDefinition.model_validate(t) for t in extracted_tools
        ]

        interactions: list[GenAiInteraction] = []
        inputs: list[Message] = []
        outputs: list[ModelResponse] = []

        def flush() -> None:
            nonlocal inputs, outputs
            if not outputs:
                return
            kept_inputs = (
                inputs[_start_index_after_last_assistant(inputs) :]
                if self.drop_redundant_input_history
                else inputs
            )
            interactions.append(
                GenAiInteraction(
                    inputs=kept_inputs,
                    outputs=outputs,
                    tool_definitions=shared_tool_definitions,
                )
            )
            inputs = []
            outputs = []

        for event in events:
            event_name, body = _event_name_and_body(event)

            if event_name in _ROLE_BY_EVENT:
                flush()
                inputs.append(_build_message(event_name, body, normalizer=normalizer))
            elif event_name == "gen_ai.choice":
                outputs.append(_build_response(body, normalizer=normalizer))
            elif self.raise_on_unknown_event:
                raise ValueError(f"Unknown event name: {event_name}")

        flush()
        if inputs:
            logger.warning(
                (
                    "Ignoring %d input message(s) without a final gen_ai.choice "
                    "(incomplete log stream or interrupted request)"
                ),
                len(inputs),
            )
        return interactions


def _build_message(
    event_name: str,
    body: dict[str, Any],
    *,
    normalizer: ProviderNormalizer,
) -> Message:
    declared_role = body.get("role") or _ROLE_BY_EVENT[event_name]
    content = body.get("content")
    tool_calls = body.get("tool_calls")
    effective_role = normalizer.derive_message_role(declared_role, content)
    parts = normalizer.parse_message_parts(
        effective_role, content, tool_calls=tool_calls
    )
    # ``gen_ai.tool.message`` carries the tool-call id at body level; copy it
    # onto the single response part the normalizer produced (if any).
    if effective_role == "tool":
        tool_id = body.get("id")
        if isinstance(tool_id, str):
            parts = [
                p.model_copy(update={"id": tool_id})
                if isinstance(p, ToolCallResponsePart) and not p.id
                else p
                for p in parts
            ]
    return Message(role=effective_role, parts=parts)  # pyright: ignore[reportArgumentType]


def _build_response(
    body: dict[str, Any],
    *,
    normalizer: ProviderNormalizer,
) -> ModelResponse:
    body = normalizer.parse_choice_body(body)
    finish_reason = body.get("finish_reason")
    index = body.get("index")
    message = body.get("message")
    if isinstance(message, dict):
        role = message.get("role", "assistant")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        parts = normalizer.parse_message_parts(role, content, tool_calls=tool_calls)
    else:
        role = "assistant"
        parts = []
    return ModelResponse(
        role=role,  # pyright: ignore[reportArgumentType]
        parts=parts,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        index=index if isinstance(index, int) else None,
    )
