from typing import Any, Literal, Self

from pydantic import BaseModel, TypeAdapter
from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.rule import Rule

from .interaction import Interaction
from .trace import Trace

ROLE_COLOR_MAPPING = {
    "system": "bold green",
    "user": "bold blue",
    "assistant": "bold yellow",
    "tool": "bold purple",
}

_DEFAULT_BORDER_STYLE = "bold gray"


class TextMessageLike(BaseModel, frozen=True, extra="allow"):
    role: str
    content: str

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(
            self.role, style=ROLE_COLOR_MAPPING.get(self.role, _DEFAULT_BORDER_STYLE)
        )
        yield self.content


class FunctionLike(BaseModel, frozen=True, extra="allow"):
    name: str
    arguments: str | dict[str, Any] | None = None

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield f"{self.name}(${repr(self.arguments)})"


class ToolCallLike(BaseModel, frozen=True, extra="allow"):
    id: str
    type: str

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield "[dim]No details available[/dim]"


class FunctionCallLike(ToolCallLike, frozen=True, extra="allow"):
    type: Literal["function_call"] = "function_call"
    function: FunctionLike

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield from self.function.__rich_console__(console, options)


class ToolCallMessageLike(BaseModel, frozen=True, extra="allow"):
    role: str
    tool_calls: list[ToolCallLike]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule("Tool calls", style=_DEFAULT_BORDER_STYLE)
        for tool_call in self.tool_calls:
            yield Panel(
                tool_call,
                title=f"{tool_call.type.capitalize()} call: {tool_call.id}",
                border_style=_DEFAULT_BORDER_STYLE,
            )


class ToolMessageLike(TextMessageLike, frozen=True, extra="allow"):
    role: str = "tool"
    id: str

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(
            f"{self.role.capitalize()}: {self.id}",
            style=ROLE_COLOR_MAPPING.get(self.role, _DEFAULT_BORDER_STYLE),
        )
        yield self.content


AssistantMessageLike = ToolCallMessageLike | TextMessageLike
MessageLike = ToolCallMessageLike | ToolMessageLike | TextMessageLike


class ChoiceLike(BaseModel, frozen=True, extra="allow"):
    message: AssistantMessageLike
    finish_reason: str | None
    index: int | None

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        title = "Choice"
        if self.index is not None:
            title += f" #{self.index}"
        if self.finish_reason is not None:
            title += f" ({self.finish_reason})"

        yield Rule(title, characters="=")
        yield from self.message.__rich_console__(console, options)


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

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if not self.interactions:
            yield "[dim]No interactions found[/dim]"
            return

        for idx, interaction in enumerate(self.interactions):
            assistant_messages = [
                assistant_message
                for assistant_message in interaction.inputs
                if assistant_message.role == "assistant"
            ]
            skip_count = min(len(assistant_messages), idx)
            for inputs in interaction.inputs:
                if skip_count > 0:
                    if inputs.role == "assistant":
                        skip_count -= 1
                    continue

                yield from inputs.__rich_console__(console, options)

            for outputs in interaction.outputs:
                yield from outputs.__rich_console__(console, options)
