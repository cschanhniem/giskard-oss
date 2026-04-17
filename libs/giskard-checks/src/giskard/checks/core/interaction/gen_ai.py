from typing import Any, Self

from pydantic import BaseModel, TypeAdapter

from .interaction import Interaction
from .trace import Trace


class TextMessageLike(BaseModel, frozen=True, extra="allow"):
    role: str
    content: str


class ToolCallLike(BaseModel, frozen=True, extra="allow"):
    id: str
    type: str


class ToolCallMessageLike(BaseModel, frozen=True, extra="allow"):
    role: str
    tool_calls: list[ToolCallLike]


class ToolMessageLike(TextMessageLike, frozen=True, extra="allow"):
    id: str


AssistantMessageLike = ToolCallMessageLike | TextMessageLike
MessageLike = ToolCallMessageLike | ToolMessageLike | TextMessageLike


class ChoiceLike(BaseModel, frozen=True, extra="allow"):
    message: AssistantMessageLike
    finish_reason: str | None
    index: int | None


_AssistantMessageLikeTypeAdapter = TypeAdapter(AssistantMessageLike)


def _body_with_role(role: str, body: dict[str, Any]) -> dict[str, Any]:
    """Role is implicit when matching the event name (genai.<role>.message)."""
    return {"role": role} | body


def _normalize_choice_body_for_semconv(body: dict[str, Any]) -> dict[str, Any]:
    """Anthropic instrumentation omits assistant ``role`` in the nested message (semconv)."""
    choice_body = dict(body)
    nested = choice_body.get("message")
    if isinstance(nested, dict) and "role" not in nested:
        choice_body["message"] = {"role": "assistant", **nested}
    return choice_body


def _event_name_and_body(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    event_name = event.get("event_name", None)
    if not isinstance(event_name, str):
        raise ValueError(f"Event name is not a string: event={event}")
    body = event.get("body", None)
    if body is None:
        raise ValueError(
            f"Event body is missing, ensure to enable OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: {event}"
        )
    if not isinstance(body, dict):
        raise ValueError(f"Event body is not a dict: {body}")
    return event_name, body


class GenAiTrace(Trace[list[MessageLike], list[ChoiceLike]], frozen=True):
    @classmethod
    def from_otel_logs(
        cls,
        events: list[dict[str, Any]],
        /,
        raise_on_unknown_event: bool = True,
    ) -> Self:
        interactions: list[Interaction[list[MessageLike], list[ChoiceLike]]] = []
        inputs: list[MessageLike] = []
        choices: list[ChoiceLike] = []

        def _flush_if_has_choices() -> None:
            nonlocal inputs, choices
            if not choices:
                return
            interactions.append(Interaction(inputs=inputs, outputs=choices))
            inputs = []
            choices = []

        for event in events:
            event_name, body = _event_name_and_body(event)

            if event_name == "gen_ai.system.message":
                _flush_if_has_choices()
                inputs.append(
                    TextMessageLike.model_validate(_body_with_role("system", body))
                )
            elif event_name == "gen_ai.user.message":
                _flush_if_has_choices()
                inputs.append(
                    TextMessageLike.model_validate(_body_with_role("user", body))
                )
            elif event_name == "gen_ai.assistant.message":
                _flush_if_has_choices()
                inputs.append(
                    _AssistantMessageLikeTypeAdapter.validate_python(
                        _body_with_role("assistant", body)
                    )
                )
            elif event_name == "gen_ai.tool.message":
                _flush_if_has_choices()
                inputs.append(
                    ToolMessageLike.model_validate(_body_with_role("tool", body))
                )
            elif event_name == "gen_ai.choice":
                choice = ChoiceLike.model_validate(
                    _normalize_choice_body_for_semconv(body)
                )
                choices.append(choice)
            elif raise_on_unknown_event:
                raise ValueError(f"Unknown event name: {event_name}")

        _flush_if_has_choices()
        if inputs:
            raise ValueError(
                f"No final choice event found after processing all events, remaining inputs: {inputs}"
            )

        return cls(interactions=interactions)
