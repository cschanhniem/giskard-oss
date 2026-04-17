"""Canonical domain model for GenAI interactions.

Aligned with the parts-based shape of OpenTelemetry GenAI semantic conventions
v1.40+ (``gen_ai.input.messages`` / ``gen_ai.output.messages``). Older v1.36
event-stream payloads are mapped into the same model by the adapters.
"""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field
from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.rule import Rule

from ..interaction import Interaction
from ..trace import Trace

ROLE_COLOR_MAPPING: dict[str, str] = {
    "system": "bold green",
    "user": "bold blue",
    "assistant": "bold yellow",
    "tool": "bold purple",
}

_DEFAULT_BORDER_STYLE = "bold gray"

Role = Literal["system", "user", "assistant", "tool"]


class TextPart(BaseModel, frozen=True):
    """Free-form text content inside a message or model response."""

    type: Literal["text"] = "text"
    content: str

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield self.content


class ToolCallPart(BaseModel, frozen=True):
    """A tool/function call requested by the model.

    ``arguments`` may be either a JSON string (as emitted by some providers) or
    a pre-parsed dict; callers should not rely on a specific form.
    """

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: str | dict[str, Any] | None = None

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield f"[bold]{self.name}[/bold]({self.arguments!r})  [dim]#{self.id}[/dim]"


class ToolCallResponsePart(BaseModel, frozen=True):
    """The result of a tool/function call, correlated by ``id``."""

    type: Literal["tool_call_response"] = "tool_call_response"
    id: str
    result: Any = None

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield f"[dim]#{self.id}[/dim] -> {self.result!r}"


Part = Annotated[
    TextPart | ToolCallPart | ToolCallResponsePart,
    Field(discriminator="type"),
]


def _render_parts(parts: list[Part]) -> RenderResult:
    for part in parts:
        yield part
        yield ""


class Message(BaseModel, frozen=True):
    """A single turn in the chat history, as sent to the model.

    The ``parts`` list is the canonical carrier of content: plain text, tool
    call requests, and tool call responses are all first-class.
    """

    role: Role
    parts: list[Part] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Concatenated text across all ``TextPart`` entries (``""`` if none)."""
        return "".join(p.content for p in self.parts if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """Tool calls contained in this message (empty if none)."""
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    @property
    def tool_responses(self) -> list[ToolCallResponsePart]:
        """Tool-call responses contained in this message (empty if none)."""
        return [p for p in self.parts if isinstance(p, ToolCallResponsePart)]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(
            self.role,
            style=ROLE_COLOR_MAPPING.get(self.role, _DEFAULT_BORDER_STYLE),
        )
        yield from _render_parts(self.parts)


class ModelResponse(BaseModel, frozen=True):
    """One model-generated choice/candidate returned for a request."""

    role: Role = "assistant"
    parts: list[Part] = Field(default_factory=list)
    finish_reason: str | None = None
    index: int | None = None

    @property
    def text(self) -> str:
        return "".join(p.content for p in self.parts if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        title = "Response"
        if self.index is not None:
            title += f" #{self.index}"
        if self.finish_reason is not None:
            title += f" ({self.finish_reason})"
        yield Rule(title, characters="=")
        yield from _render_parts(self.parts)


class ToolDefinition(BaseModel, frozen=True, extra="allow"):
    """Description of a tool the model is allowed to call.

    ``extra="allow"`` preserves provider-specific fields (e.g. ``description``,
    ``strict``) without requiring an exhaustive schema in this library.
    """

    type: str = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class GenAiInteraction(
    Interaction[list[Message], list[ModelResponse]],
    frozen=True,
):
    """One request/response pair with system-level context.

    Extends :class:`Interaction` with GenAI-specific side channels:
    ``system_instructions`` (per ``gen_ai.system_instructions`` semconv) and
    ``tool_definitions`` (per ``gen_ai.tool.definitions`` semconv). These
    carry context that is *about* the interaction but is not itself a message.
    """

    system_instructions: list[TextPart] = Field(default_factory=list)
    tool_definitions: list[ToolDefinition] = Field(default_factory=list)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if self.system_instructions:
            yield Rule("System instructions", style=ROLE_COLOR_MAPPING["system"])
            yield from _render_parts(list(self.system_instructions))
        for message in self.inputs:
            yield from message.__rich_console__(console, options)
        for response in self.outputs:
            yield from response.__rich_console__(console, options)


class GenAiTrace(
    Trace[list[Message], list[ModelResponse], GenAiInteraction],
    frozen=True,
):
    """Trace of GenAI interactions parsed from OpenTelemetry telemetry.

    Use the :meth:`from_otel`, :meth:`from_event_stream` or
    :meth:`from_span_attributes` classmethods to build instances from OTel
    sources. See :mod:`.adapters` for the parsing strategies.
    """

    @classmethod
    def from_otel(
        cls,
        source: Any,
        /,
        *,
        provider: str | None = None,
        raise_on_unknown_event: bool = True,
        drop_redundant_input_history: bool = False,
    ) -> Self:
        """Parse an OTel GenAI payload, auto-detecting the semconv family.

        Parameters
        ----------
        source
            Either a list of event dicts (semconv ≤1.36, ``{event_name, body}``)
            or an attributes dict from a single inference span
            (semconv ≥1.40, ``gen_ai.client.inference.operation.details``).
        provider
            ``gen_ai.provider.name`` / ``gen_ai.system`` (e.g. ``"openai"``,
            ``"anthropic"``). Auto-detected from the source when omitted; pass
            it explicitly to force a provider-specific normalizer.
        raise_on_unknown_event
            Event-stream only: if ``True`` (default), raise on unknown event
            names instead of silently skipping them.
        drop_redundant_input_history
            Event-stream only: some instrumentations repeat the full
            conversation in the message stream before every completion; when
            ``True`` the adapter keeps only messages after the last
            ``assistant`` entry in each interaction's inputs.
        """
        from .adapters import parse_source

        interactions = parse_source(
            source,
            provider=provider,
            raise_on_unknown_event=raise_on_unknown_event,
            drop_redundant_input_history=drop_redundant_input_history,
        )
        return cls(interactions=list(interactions))

    @classmethod
    def from_event_stream(
        cls,
        events: list[dict[str, Any]],
        /,
        *,
        provider: str | None = None,
        raise_on_unknown_event: bool = True,
        drop_redundant_input_history: bool = False,
    ) -> Self:
        """Parse semconv ≤1.36 GenAI event logs (``gen_ai.user.message``, …)."""
        from .adapters.events import EventStreamAdapter
        from .providers import get_normalizer

        adapter = EventStreamAdapter(
            raise_on_unknown_event=raise_on_unknown_event,
            drop_redundant_input_history=drop_redundant_input_history,
        )
        interactions = adapter.parse(events, normalizer=get_normalizer(provider))
        return cls(interactions=interactions)

    @classmethod
    def from_span_attributes(
        cls,
        attributes: dict[str, Any],
        /,
        *,
        provider: str | None = None,
    ) -> Self:
        """Parse semconv ≥1.40 ``gen_ai.client.inference.operation.details``."""
        from .adapters.attributes import SpanAttributesAdapter
        from .providers import get_normalizer

        adapter = SpanAttributesAdapter()
        interactions = adapter.parse(attributes, normalizer=get_normalizer(provider))
        return cls(interactions=interactions)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if not self.interactions:
            yield "[dim]No interactions found[/dim]"
            return
        for idx, interaction in enumerate(self.interactions):
            yield Panel(
                interaction,
                title=f"Interaction {idx + 1}",
                border_style=_DEFAULT_BORDER_STYLE,
            )
