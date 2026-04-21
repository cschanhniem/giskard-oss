from typing import Any, Literal

from pydantic import BaseModel, field_validator

from ._validation import NotEmptyList, NotEmptyStr, JsonSchema

# -- Tool types -------------------------------------------------------

class FunctionDefinition(BaseModel):
    name: NotEmptyStr
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """
        The name of the function to be called. Must be a-z, A-Z, 0-9, or contain
        underscores and dashes, with a maximum length of 64.
        """
        import re

        if len(v) > 64:
            raise ValueError("Function name must be at most 64 characters long.")
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Function name must only contain letters, numbers, underscores, or dashes."
            )
        return v

class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition

class Function(BaseModel):
    name: NotEmptyStr
    arguments: dict[str, Any]


class FunctionCall(BaseModel):
    type: Literal["function_call"] = "function_call"
    id: NotEmptyStr
    function_call: Function


# -- Content part types --------------------------------------------------------


class TextContentPart(BaseModel):
    type: Literal["text"] = "text"
    text: NotEmptyStr


class ImageURL(BaseModel):
    url: NotEmptyStr


class ImageURLContentPart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


# -- Message types -------------------------------------------------------------


class SystemMessage(BaseModel):
    role: Literal["system", "developer"] = "system"
    content: NotEmptyStr | NotEmptyList[TextContentPart]


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: NotEmptyStr | NotEmptyList[TextContentPart | ImageURLContentPart]

    @field_validator("content", mode="after")
    def _at_least_one_text_or_image_url_part(
        self, v: NotEmptyStr | NotEmptyList[TextContentPart | ImageURLContentPart]
    ) -> NotEmptyStr | NotEmptyList[TextContentPart | ImageURLContentPart]:
        if isinstance(v, str):
            return v

        if not any(isinstance(part, TextContentPart) for part in v):
            raise ValueError("At least one text part is required")

        return v


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: NotEmptyStr | NotEmptyList[TextContentPart] | None

class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: NotEmptyStr | NotEmptyList[TextContentPart]
    tool_call_id: NotEmptyStr


class ChatParameters(BaseModel, extra="ignore"):
    messages: NotEmptyList[
        SystemMessage | UserMessage | AssistantMessage | ToolMessage
    ]
    tools: list[FunctionTool] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    metadata: dict[str, Any] | None = None
    response_format: JsonSchema | None = None

    @field_validator("messages", mode="after")
    def _validate_messages(self, v: NotEmptyList[SystemMessage | UserMessage | AssistantMessage | ToolMessage]) -> NotEmptyList[SystemMessage | UserMessage | AssistantMessage | ToolMessage]:
        if not any(m.role == "system" for m in v):
            raise ValueError("At least one system message is required")

        return v