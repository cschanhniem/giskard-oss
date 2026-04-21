from typing import Any, Literal, Required, TypedDict

from pydantic import BaseModel

# -- Tool types -------------------------------------------------------

class FunctionDefinition(TypedDict, total=False):
    name: Required[str]
    description: str
    parameters: dict[str, Any]
    strict: bool


class FunctionTool(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[FunctionDefinition]

class Function(TypedDict, total=False):
    name: Required[str]
    arguments: Required[dict[str, Any]]


class FunctionCall(TypedDict, total=False):
    type: Required[Literal["function_call"]]
    id: Required[str]
    function_call: Required[Function]


# -- Content types -------------------------------------------------------------


class TextContent(TypedDict, total=False):
    type: Required[Literal["text"]]
    text: Required[str]


class ImageURL(TypedDict, total=False):
    url: Required[str]
    detail: Literal["auto", "low", "high"]


class ImageURLContent(TypedDict, total=False):
    type: Required[Literal["image_url"]]
    image_url: Required[ImageURL]


Content = TextContent | ImageURLContent

# -- Message types -------------------------------------------------------------


class SystemMessage(TypedDict, total=False):
    role: Required[Literal["system", "developer"]]
    content: Required[str | list[TextContent]]


class UserMessage(TypedDict, total=False):
    role: Required[Literal["user"]]
    content: Required[str | list[Content]]


class AssistantMessage(TypedDict, total=False):
    role: Required[Literal["assistant"]]
    content: str | list[TextContent]
    tool_calls: list[FunctionCall]


class ToolMessage(TypedDict, total=False):
    role: Required[Literal["tool"]]
    content: Required[str | list[TextContent]]
    tool_call_id: Required[str]

ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage

# -- Chat Completion types -----------------------------------------------------

class CompletionFunction(BaseModel, extra="allow"):
    name: str
    arguments: dict[str, Any]

class CompletionFunctionToolCall(BaseModel, extra="allow"):
    type: Literal["function"] = "function"
    id: str
    function: CompletionFunction

class CompletionMessage(BaseModel, extra="allow"):
    role: str | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[CompletionFunctionToolCall] | None = None


class CompletionChoice(BaseModel, extra="allow"):
    message: CompletionMessage
    finish_reason: str | None = None
    index: int = 0


class CompletionUsage(BaseModel, extra="allow"):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel, extra="allow"):
    choices: list[CompletionChoice]
    model: str | None = None
    usage: CompletionUsage | None = None
