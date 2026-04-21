from typing import Any, Literal

from pydantic import BaseModel, model_validator

from ._validation import NotEmptyList, NotEmptyStr


class Function(BaseModel):
    type: Literal["function"] = "function"
    name: NotEmptyStr
    parameters: dict[str, Any]
    description: str | None = None

    @model_validator(mode="before")
    def _handle_chat_to_function(self, v: Any) -> Any:
        if (
            isinstance(v, dict)
            and v.get("type") == "function"
            and isinstance(v.get("function"), dict)
        ):
            return {"type": "function", **v["function"]}
        return v


class InputText(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: NotEmptyStr


class InputImage(BaseModel):
    type: Literal["input_image"] = "input_image"
    details: Literal["auto", "low", "high", "original"] = "auto"
    file_id: NotEmptyStr | None = None
    image_url: NotEmptyStr | None = None

    @model_validator(mode="after")
    def validate_file_id_or_image_url(self) -> "InputImage":
        if self.file_id is None and self.image_url is None:
            raise ValueError("Either file_id or image_url must be provided")
        if self.file_id is not None and self.image_url is not None:
            raise ValueError("Only one of file_id or image_url must be provided")
        return self


class EasyInputMessage(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system", "developer"]
    content: NotEmptyStr | NotEmptyList[InputText | InputImage]


class ResponseParameters(BaseModel, extra="ignore"):
    model: NotEmptyStr
    input: str | list[EasyInputMessage]
    instructions: str | None = None
    previous_response_id: str | None = None
    tools: list[Function] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
