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
from .providers.base import CompletionProvider, EmbeddingProvider, ResponseProvider
from .retry import should_retry
from .routing import LLMClient, acompletion, aembedding, aresponse, configure, reset
from .types import (
    AssistantMessage,
    ChatCompletion,
    Choice,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    Function,
    FunctionCall,
    FunctionDef,
    FunctionMessage,
    FunctionTool,
    FunctionToolDefinition,
    InputMessage,
    Message,
    RefusalContent,
    ResponseOutputFunctionCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseResult,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    assistant,
    developer,
    system,
    tool,
    tool_calls,
    user,
)

__all__ = [
    # Functions
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
    # Types — Chat Completion
    "ChatCompletion",
    "Choice",
    "Usage",
    # Types — Messages
    "Message",
    "InputMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "FunctionMessage",
    # Types — Content parts
    "TextContent",
    "RefusalContent",
    # Types — Tools
    "Tool",
    "FunctionTool",
    "FunctionToolDefinition",
    "FunctionDef",
    # Types — Tool calls
    "ToolCall",
    "FunctionCall",
    "Function",
    # Types — Embedding
    "EmbeddingResponse",
    "EmbeddingData",
    "EmbeddingUsage",
    # Types — Response
    "ResponseResult",
    "ResponseOutputMessage",
    "ResponseOutputFunctionCall",
    "ResponseOutputItem",
    # Message builders
    "user",
    "system",
    "developer",
    "assistant",
    "tool",
    "tool_calls",
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
