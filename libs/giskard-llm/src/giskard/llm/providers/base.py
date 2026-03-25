"""Abstract base for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any

from ..types import ChatMessage, CompletionResponse, EmbeddingResponse


class BaseProvider(ABC):
    """Each provider translates between the unified API and its native SDK."""

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        **params: Any,
    ) -> CompletionResponse:
        """Send a chat completion request and return a unified response."""

    @abstractmethod
    async def embed(
        self,
        model: str,
        input: list[str],
        **params: Any,
    ) -> EmbeddingResponse:
        """Generate embeddings and return a unified response."""
