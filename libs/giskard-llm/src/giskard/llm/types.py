"""Unified response and message types for giskard-llm.

All types are Pydantic models with discriminated unions where appropriate.
Messages, tools, content parts, tool calls, and response items follow an
OpenAI-shaped canonical schema that providers translate to/from their own
wire formats.
"""

from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    ValidationError,
    model_validator,
)

# -- Base model --------------------------------------------------


class _BaseModel(BaseModel):
    """Shared base for all giskard-llm response models. Defaults model_dump to exclude None fields."""

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(**kwargs)


# -- Discriminator helpers ---------------------------------------

_UNKNOWN_TAG = "__unknown__"


def _type_discriminator(known: set[str]) -> Callable[[Any], str]:
    """Build a callable discriminator that routes known ``type`` values to their
    concrete branch and everything else to the ``Unknown*`` fallback.

    Pydantic's ``Field(discriminator=...)`` requires every variant's discriminator
    field to be a ``Literal``. Our ``Unknown*`` branches intentionally accept any
    string, so we use a callable ``Discriminator`` + ``Tag`` instead.
    """

    def _discriminate(value: Any) -> str:
        if isinstance(value, dict):
            t = value.get("type")
        else:
            t = getattr(value, "type", None)
        return t if isinstance(t, str) and t in known else _UNKNOWN_TAG

    return _discriminate


# -- Tool definition types ---------------------------------------


class FunctionDef(_BaseModel):
    """Function schema without a redundant ``type`` field.

    Used as the inner payload of :class:`FunctionToolDefinition` (the nested
    OpenAI Chat Completions tool wire format).
    """

    name: str
    description: str
    parameters: dict[str, Any]


class FunctionTool(_BaseModel):
    """Flat function tool (type/name/description/parameters).

    Mirrors the OpenAI Responses API tool wire format.
    """

    type: Literal["function"] = "function"
    name: str
    description: str
    parameters: dict[str, Any]


class FunctionToolDefinition(_BaseModel):
    """Nested function tool definition ``{type, function: {name, description, parameters}}``.

    Mirrors the OpenAI Chat Completions tool wire format.
    """

    type: Literal["function"] = "function"
    function: FunctionDef


type ToolInput = FunctionTool | FunctionToolDefinition | dict[str, Any]


class UnknownTool(_BaseModel, extra="allow"):
    type: str


Tool = Annotated[
    Annotated[FunctionTool, Tag("function")]
    | Annotated[UnknownTool, Tag(_UNKNOWN_TAG)],
    Discriminator(_type_discriminator({"function"})),
]

# -- Tool call types ---------------------------------------


class Function(_BaseModel):
    name: str
    arguments: str


class FunctionCall(_BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: Function


class UnknownCall(_BaseModel, extra="allow"):
    type: str


ToolCall = Annotated[
    Annotated[FunctionCall, Tag("function")]
    | Annotated[UnknownCall, Tag(_UNKNOWN_TAG)],
    Discriminator(_type_discriminator({"function"})),
]

# -- Message content types ---------------------------------------


class TextContent(_BaseModel):
    type: Literal["text"] = "text"
    text: str


class RefusalContent(_BaseModel):
    type: Literal["refusal"] = "refusal"
    text: str


class UnknownContent(_BaseModel, extra="allow"):
    type: str


def _coerce_text_content_parts(
    value: str | list[str | dict[str, Any]],
) -> list[dict[str, Any]]:
    def _coerce_part(part: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(part, str):
            return {"type": "text", "text": part}
        return part

    if isinstance(value, str):
        return [_coerce_part(value)]

    return [_coerce_part(part) for part in value]


T = TypeVar("T", bound=BaseModel)
ContentParts = Annotated[
    list[T | str] | str, BeforeValidator(_coerce_text_content_parts)
]

InputContent = Annotated[
    Annotated[TextContent, Tag("text")] | Annotated[UnknownContent, Tag(_UNKNOWN_TAG)],
    Discriminator(_type_discriminator({"text"})),
]
OutputContent = Annotated[
    Annotated[TextContent, Tag("text")]
    | Annotated[RefusalContent, Tag("refusal")]
    | Annotated[UnknownContent, Tag(_UNKNOWN_TAG)],
    Discriminator(_type_discriminator({"text", "refusal"})),
]

# -- Message types ---------------------------------------


class InputMessage(_BaseModel):
    role: Literal["user"] = "user"
    content: ContentParts[InputContent]
    name: str | None = None

    @property
    def text(self) -> str | None:
        if self.content is None:
            return None

        if isinstance(self.content, str):
            return self.content

        texts = [part.text for part in self.content if isinstance(part, TextContent)]
        return "\n".join(texts) if texts else None


class SystemMessage(_BaseModel):
    role: Literal["system", "developer"] = "system"
    content: ContentParts[TextContent]
    name: str | None = None

    @property
    def text(self) -> str | None:
        if self.content is None:
            return None

        if isinstance(self.content, str):
            return self.content

        texts = [part.text for part in self.content if isinstance(part, TextContent)]
        return "\n".join(texts) if texts else None


class AssistantMessage(_BaseModel):
    role: Literal["assistant"] = "assistant"
    content: ContentParts[OutputContent] | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_refusal(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "refusal" not in data:
            return data

        refusal = data.pop("refusal")
        if refusal is None:
            return data
        if isinstance(refusal, str):
            refusal = {"type": "refusal", "text": refusal}
        else:
            try:
                refusal = RefusalContent.model_validate(refusal)
            except ValidationError as e:
                raise ValueError("Invalid refusal content") from e

        if data.get("content") is not None and data.get("content") != []:
            raise ValueError("Content and refusal cannot be mixed")

        data["content"] = [refusal]
        return data

    @property
    def text(self) -> str | None:
        if self.content is None:
            return None

        if isinstance(self.content, str):
            return self.content

        texts = [part.text for part in self.content if isinstance(part, TextContent)]
        return "\n".join(texts) if texts else None

    @property
    def refusal(self) -> str | None:
        if self.content is None:
            return None

        refusals = [part for part in self.content if isinstance(part, RefusalContent)]
        if not refusals:
            return None

        return "\n".join([refusal.text for refusal in refusals])


class ToolMessage(_BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str | None = None
    name: str | None = None
    content: ContentParts[TextContent]

    @property
    def text(self) -> str | None:
        if isinstance(self.content, str):
            return self.content

        texts = [part.text for part in self.content if isinstance(part, TextContent)]
        return "\n".join(texts) if texts else None


class FunctionMessage(_BaseModel):
    role: Literal["function"] = "function"
    name: str
    content: str | None = None


Message = Annotated[
    InputMessage | SystemMessage | AssistantMessage | ToolMessage | FunctionMessage,
    Field(discriminator="role"),
]


# -- Choice types (Chat Completions shape) ---------------------


class Choice(_BaseModel):
    message: AssistantMessage
    finish_reason: str | None = None
    index: int = 0


class Usage(_BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(_BaseModel):
    id: str | None = None
    choices: list[Choice]
    model: str | None = None
    usage: Usage | None = None


# -- Response types (Responses API shape) ---------------------------------------


class ResponseOutputMessage(_BaseModel):
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: ContentParts[OutputContent]


class ResponseOutputFunctionCall(_BaseModel):
    type: Literal["function_call"] = "function_call"
    call_id: str | None = None
    name: str
    arguments: dict[str, Any]


class ResponseOutputUnknown(_BaseModel, extra="allow"):
    type: str


ResponseOutputItem = Annotated[
    Annotated[ResponseOutputMessage, Tag("message")]
    | Annotated[ResponseOutputFunctionCall, Tag("function_call")]
    | Annotated[ResponseOutputUnknown, Tag(_UNKNOWN_TAG)],
    Discriminator(_type_discriminator({"message", "function_call"})),
]


class ResponseResult(_BaseModel):
    id: str
    outputs: list[ResponseOutputItem]
    model: str | None = None
    usage: Usage | None = None

    @property
    def output_text(self) -> str | None:
        if isinstance(self.outputs, str):
            return self.outputs

        texts = [
            content.text
            for output in self.outputs
            if isinstance(output, ResponseOutputMessage)
            for content in output.content
            if isinstance(content, TextContent)
        ]
        if not texts:
            return None
        return "\n".join(texts)

    @property
    def function_calls(self) -> list[ResponseOutputFunctionCall]:
        if not self.outputs:
            return []

        return [
            call
            for call in self.outputs
            if isinstance(call, ResponseOutputFunctionCall)
        ]

    @property
    def refusals(self) -> str | None:
        if not self.outputs:
            return None

        texts = [
            content.text
            for output in self.outputs
            if isinstance(output, ResponseOutputMessage)
            for content in output.content
            if isinstance(content, RefusalContent)
        ]
        if not texts:
            return None
        return "\n".join(texts)


# -- Embedding types ---------------------------------------


class EmbeddingData(_BaseModel):
    embedding: list[float]
    index: int


class EmbeddingUsage(_BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(_BaseModel):
    data: list[EmbeddingData] = Field(default_factory=list)
    model: str | None = None
    usage: EmbeddingUsage | None = None


# -- Helper functions ---------------------------------------


def user(*content: InputContent | str, name: str | None = None) -> InputMessage:
    return InputMessage(role="user", content=list(content), name=name)


def system(*content: TextContent | str, name: str | None = None) -> SystemMessage:
    return SystemMessage(role="system", content=list(content), name=name)


def developer(*content: TextContent | str, name: str | None = None) -> SystemMessage:
    return SystemMessage(role="developer", content=list(content), name=name)


def assistant(
    *content: OutputContent | str,
    name: str | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        role="assistant", content=list(content), name=name, tool_calls=tool_calls
    )


def tool_calls(*tool_calls: ToolCall, name: str | None = None) -> AssistantMessage:
    return AssistantMessage(tool_calls=list(tool_calls), name=name)


def tool(
    *content: TextContent | str, tool_call_id: str, name: str | None = None
) -> ToolMessage:
    return ToolMessage(
        role="tool", content=list(content), tool_call_id=tool_call_id, name=name
    )


def fn_tool_definition(
    name: str, description: str, parameters: dict[str, Any]
) -> FunctionToolDefinition:
    return FunctionToolDefinition(
        type="function",
        function=FunctionDef(name=name, description=description, parameters=parameters),
    )


def _tool_definition_from_flat_tool(tool: FunctionTool) -> FunctionToolDefinition:
    return FunctionToolDefinition(
        type=tool.type,
        function=FunctionDef(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
        ),
    )


def _flat_tool_from_definition(tool: FunctionToolDefinition) -> FunctionTool:
    return FunctionTool(
        type=tool.type,
        name=tool.function.name,
        description=tool.function.description,
        parameters=tool.function.parameters,
    )


def _coerce_tool_definition(tool: ToolInput) -> FunctionToolDefinition:
    if isinstance(tool, FunctionToolDefinition):
        return tool
    if isinstance(tool, FunctionTool):
        return _tool_definition_from_flat_tool(tool)
    if "function" in tool:
        return FunctionToolDefinition.model_validate(tool)
    return _tool_definition_from_flat_tool(FunctionTool.model_validate(tool))


def _coerce_response_tool(tool: ToolInput) -> FunctionTool:
    if isinstance(tool, FunctionTool):
        return tool
    if isinstance(tool, FunctionToolDefinition):
        return _flat_tool_from_definition(tool)
    if "function" in tool:
        return _flat_tool_from_definition(FunctionToolDefinition.model_validate(tool))
    return FunctionTool.model_validate(tool)


# -- Validation / serialization helpers ---------------------------------------


def flatten_text_content(value: Any) -> Any:
    """Collapse text-only content part lists to a plain string."""
    if not isinstance(value, list):
        return value
    if not value:
        return ""

    texts: list[str] = []
    for part in value:
        if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
            texts.append(part["text"])
        else:
            return value

    return "\n".join(texts) if texts else ""


def serialize_message(
    message: Message, *, flatten_text: bool = False
) -> dict[str, Any]:
    """Serialize a validated message to an OpenAI-shaped dict."""
    dumped = message.model_dump()
    if flatten_text and "content" in dumped and dumped["content"] is not None:
        dumped["content"] = flatten_text_content(dumped["content"])
    return dumped


def serialize_messages(
    messages: Sequence[Message], *, flatten_text: bool = False
) -> list[dict[str, Any]]:
    return [
        serialize_message(message, flatten_text=flatten_text) for message in messages
    ]


# -- Type adapters ---------------------------------------

_MESSAGE_LIST_ADAPTER: TypeAdapter[list[Message]] = TypeAdapter(list[Message])


def validate_messages(*messages: Message | dict[str, Any]) -> list[Message]:
    return _MESSAGE_LIST_ADAPTER.validate_python(list(messages))


def validate_tools(
    *tools: ToolInput,
) -> list[FunctionToolDefinition]:
    return [_coerce_tool_definition(tool) for tool in tools]


def validate_response_tools(
    *tools: ToolInput,
) -> list[FunctionTool]:
    return [_coerce_response_tool(tool) for tool in tools]
