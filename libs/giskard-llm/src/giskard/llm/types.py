"""Response types for giskard-llm.

These mirror the OpenAI-style response shapes that litellm used,
so existing code in giskard-agents can consume them with minimal changes.
"""

from typing import Any

from pydantic import BaseModel, Field


class ChoiceMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return super().model_dump(exclude_none=True, **kwargs)


class Choice(BaseModel):
    message: ChoiceMessage
    finish_reason: str | None = None
    index: int = 0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    choices: list[Choice]
    model: str | None = None
    usage: Usage | None = None


class EmbeddingData(BaseModel):
    embedding: list[float]
    index: int = 0


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingData] = Field(default_factory=list)
    model: str | None = None
    usage: EmbeddingUsage | None = None
