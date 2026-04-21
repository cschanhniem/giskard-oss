from pydantic import BaseModel, Field


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
