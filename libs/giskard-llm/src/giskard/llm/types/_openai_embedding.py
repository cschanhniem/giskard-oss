from pydantic import BaseModel


class EmbeddingParameters(BaseModel, extra="ignore"):
    model: str
    input: list[str]
    dimensions: int | None = None
