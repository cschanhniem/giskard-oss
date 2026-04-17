"""giskard-llm -- lightweight LLM routing over native provider SDKs."""

from .errors import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    LLMTimeoutError,
    ProviderNotAvailableError,
    RateLimitError,
    ServerError,
    UnsupportedOperationError,
)
from .otel import instrumentor_for_provider
from .providers.base import CompletionProvider, EmbeddingProvider, ResponseProvider
from .retry import should_retry
from .routing import LLMClient, acompletion, aembedding, aresponse, configure, reset
from .types import (
    ChatMessage,
    Choice,
    ChoiceMessage,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCallOutput,
    FunctionDef,
    ResponseOutputFunctionCall,
    ResponseOutputItem,
    ResponseOutputText,
    ResponseResult,
    ToolCall,
    ToolCallFunction,
    ToolDef,
    Usage,
)

__all__ = [
    # Functions / OTEL helpers
    "instrumentor_for_provider",
    "acompletion",
    "aembedding",
    "aresponse",
    "configure",
    "reset",
    "should_retry",
    # Client
    "LLMClient",
    # Protocols
    "CompletionProvider",
    "EmbeddingProvider",
    "ResponseProvider",
    # Types — Completion
    "CompletionResponse",
    "Choice",
    "ChoiceMessage",
    "ChatMessage",
    "Usage",
    # Types — Tools
    "ToolCall",
    "ToolCallFunction",
    "ToolDef",
    "FunctionDef",
    "FunctionCallOutput",
    # Types — Embedding
    "EmbeddingResponse",
    "EmbeddingData",
    "EmbeddingUsage",
    # Types — Response
    "ResponseResult",
    "ResponseOutputText",
    "ResponseOutputFunctionCall",
    "ResponseOutputItem",
    # Errors
    "LLMError",
    "AuthenticationError",
    "BadRequestError",
    "RateLimitError",
    "ServerError",
    "LLMTimeoutError",
    "UnsupportedOperationError",
    "ProviderNotAvailableError",
]
