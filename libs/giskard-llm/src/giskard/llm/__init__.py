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
from .routing import LLMClient
from .routing import route_completion as acompletion
from .routing import route_embedding as aembedding
from .types import (
    ChatMessage,
    Choice,
    ChoiceMessage,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    ToolCall,
    ToolCallFunction,
)

__all__ = [
    "LLMClient",
    "acompletion",
    "aembedding",
    "should_retry",
    "CompletionResponse",
    "Choice",
    "ChoiceMessage",
    "ToolCall",
    "ToolCallFunction",
    "ChatMessage",
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
