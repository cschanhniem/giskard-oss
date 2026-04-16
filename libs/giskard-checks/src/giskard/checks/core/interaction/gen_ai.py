from typing import Any, Self

from pydantic import BaseModel, TypeAdapter

from .trace import Interaction, Trace


class TextMessageLike(BaseModel, frozen=True, extra="allow"):
    role: str
    content: str


class ToolCallLike(BaseModel, frozen=True, extra="allow"):
    role: str
    id: str
    type: str


class ToolCallMessageLike(BaseModel, frozen=True, extra="allow"):
    role: str
    tool_calls: list[ToolCallLike]


class ToolMessageLike(TextMessageLike, frozen=True, extra="allow"):
    id: str


AssistantMessageLike = ToolCallMessageLike | TextMessageLike
_AssistantMessageLikeTypeAdapter = TypeAdapter(AssistantMessageLike)
MessageLike = ToolCallMessageLike | ToolMessageLike | TextMessageLike


class ChoiceLike(BaseModel, frozen=True, extra="allow"):
    message: AssistantMessageLike
    finish_reason: str | None
    index: int | None


class GenAiTrace(Trace[list[MessageLike], ChoiceLike], frozen=True):
    @classmethod
    def from_otel_logs(
        cls,
        events: list[dict[str, Any]],
        /,
        raise_on_unknown_event: bool = True,
    ) -> Self:
        interactions = []
        inputs = []
        for event in events:
            event_name = event.get("event_name", None)
            body = event.get("body", None)

            if not isinstance(event_name, str):
                raise ValueError(f"Event name is not a string: event={event}")

            if body is None:
                raise ValueError(
                    f"Event body is missing, ensure to enable OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: {event}"
                )
            if not isinstance(body, dict):
                raise ValueError(f"Event body is not a dict: {body}")

            if event_name == "gen_ai.system.message":
                body = {"role": "system"} | body  # role is optional in the body
                inputs.append(TextMessageLike.model_validate(body))
            elif event_name == "gen_ai.user.message":
                body = {"role": "user"} | body  # role is optional in the body
                inputs.append(TextMessageLike.model_validate(body))
            elif event_name == "gen_ai.assistant.message":
                body = {"role": "assistant"} | body  # role is optional in the body
                inputs.append(_AssistantMessageLikeTypeAdapter.validate_python(body))
            elif event_name == "gen_ai.tool.message":
                body = {"role": "tool"} | body  # role is optional in the body
                inputs.append(ToolMessageLike.model_validate(body))
            elif event_name == "gen_ai.choice":
                # Anthropic instrumentation omits assistant ``role`` in the message body (semconv).
                choice_body = dict(body)
                nested = choice_body.get("message")
                if isinstance(nested, dict) and "role" not in nested:
                    choice_body["message"] = {"role": "assistant", **nested}
                choice = ChoiceLike.model_validate(choice_body)
                interactions.append(Interaction(inputs=inputs, outputs=choice))
                inputs = []
            elif raise_on_unknown_event:
                raise ValueError(f"Unknown event name: {event_name}")

        if inputs:
            raise ValueError(
                f"No final choice event found after processing all events, remaining inputs: {inputs}"
            )

        return cls(interactions=interactions)
