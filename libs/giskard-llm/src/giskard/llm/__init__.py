"""giskard-llm -- lightweight LLM routing over native provider SDKs."""

from .errors import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    ProviderNotAvailableError,
    RateLimitError,
    ServerError,
    TimeoutError,
)
from .retry import should_retry
from .routing import route_completion as acompletion
from .routing import route_embedding as aembedding
from .types import (
    Choice,
    ChoiceMessage,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
)

__all__ = [
    "acompletion",
    "aembedding",
    "should_retry",
    "CompletionResponse",
    "Choice",
    "ChoiceMessage",
    "EmbeddingResponse",
    "EmbeddingData",
    "LLMError",
    "AuthenticationError",
    "BadRequestError",
    "RateLimitError",
    "ServerError",
    "TimeoutError",
    "ProviderNotAvailableError",
]
