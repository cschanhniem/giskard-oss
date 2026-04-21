from typing import Any, Literal, Required, TypedDict

from pydantic import BaseModel, Field


class InputText(TypedDict, total=False):
    type: Required[Literal["input_text"]]
    text: Required[str]


class InputImage(TypedDict, total=False):
    type: Required[Literal["input_image"]]
    details: Literal["auto", "low", "high", "original"]
    file_id: str
    image_url: str


class EasyInputMessage(TypedDict, total=False):
    role: Required[Literal["user", "assistant", "system", "developer"]]
    content: Required[str | list[InputText | InputImage]]
    type: Literal["message"]


class ResponseOutputText(BaseModel, extra="allow"):
    type: Literal["text"] = "text"
    text: str


class ResponseOutputRefusal(BaseModel, extra="allow"):
    type: Literal["refusal"] = "refusal"
    refusal: str


class ResponseOutputFunctionCall(BaseModel, extra="allow"):
    type: Literal["function_call"] = "function_call"
    call_id: str | None = None
    name: str
    arguments: dict[str, Any]


class ResponseOutputMessage(BaseModel, extra="allow"):
    type: Literal["message"] = "message"
    role: str | None = None
    content: str | list[ResponseOutputText | ResponseOutputRefusal]


class FunctionCall(BaseModel, extra="allow"):
    type: Literal["function_call"] = "function_call"
    call_id: str | None = None
    name: str
    arguments: dict[str, Any]


class ResponseUsage(BaseModel, extra="allow"):
    input_token: int = 0
    output_token: int = 0
    total_tokens: int = 0


class ResponseResult(BaseModel, extra="allow"):
    id: str
    output: list[ResponseOutputMessage | FunctionCall] = Field(
        validation_alias="outputs"
    )
    model: str | None = None
    usage: ResponseUsage | None = None
