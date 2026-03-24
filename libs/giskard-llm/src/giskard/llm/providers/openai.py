"""OpenAI provider -- uses the ``openai`` SDK directly."""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportImplicitRelativeImport=false

from typing import Any

from ..errors import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    ProviderNotAvailableError,
    RateLimitError,
    ServerError,
    TimeoutError,
)
from ..types import (
    Choice,
    ChoiceMessage,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    Usage,
)
from .base import BaseProvider

PROVIDER = "openai"


def _import_openai():
    try:
        import openai

        return openai
    except ImportError:
        raise ProviderNotAvailableError(PROVIDER, "openai")


class OpenAIProvider(BaseProvider):
    def __init__(self) -> None:
        openai = _import_openai()
        self._client = openai.AsyncOpenAI()

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **params: Any,
    ) -> CompletionResponse:
        openai = _import_openai()
        kwargs = self._build_completion_kwargs(model, messages, params)
        try:
            raw = await self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            raise RateLimitError(429, str(e), PROVIDER) from e
        except openai.AuthenticationError as e:
            raise AuthenticationError(e.status_code, str(e), PROVIDER) from e
        except openai.BadRequestError as e:
            raise BadRequestError(e.status_code, str(e), PROVIDER) from e
        except openai.APITimeoutError as e:
            raise TimeoutError(408, str(e), PROVIDER) from e
        except openai.InternalServerError as e:
            raise ServerError(e.status_code, str(e), PROVIDER) from e
        except openai.APIError as e:
            raise LLMError(e.status_code or 500, str(e), PROVIDER) from e

        return self._to_completion_response(raw)

    async def embed(
        self,
        model: str,
        input: list[str],
        **params: Any,
    ) -> EmbeddingResponse:
        openai = _import_openai()
        kwargs: dict[str, Any] = {"model": model, "input": input}
        if "dimensions" in params and params["dimensions"] is not None:
            kwargs["dimensions"] = params["dimensions"]
        try:
            raw = await self._client.embeddings.create(**kwargs)
        except openai.RateLimitError as e:
            raise RateLimitError(429, str(e), PROVIDER) from e
        except openai.AuthenticationError as e:
            raise AuthenticationError(e.status_code, str(e), PROVIDER) from e
        except openai.APITimeoutError as e:
            raise TimeoutError(408, str(e), PROVIDER) from e
        except openai.APIError as e:
            raise LLMError(e.status_code or 500, str(e), PROVIDER) from e

        return self._to_embedding_response(raw)

    # -- helpers ---------------------------------------------------------------

    def _build_completion_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model, "messages": messages}

        if params.get("temperature") is not None:
            kwargs["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            kwargs["max_tokens"] = params["max_tokens"]
        if params.get("timeout") is not None:
            kwargs["timeout"] = params["timeout"]
        if params.get("tools"):
            kwargs["tools"] = params["tools"]
        if params.get("metadata"):
            kwargs["metadata"] = params["metadata"]

        response_format = params.get("response_format")
        if response_format is not None:
            kwargs["response_format"] = response_format

        return kwargs

    def _to_completion_response(self, raw: Any) -> CompletionResponse:
        choices = []
        for c in raw.choices:
            tool_calls = None
            if c.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in c.message.tool_calls
                ]
            choices.append(
                Choice(
                    message=ChoiceMessage(
                        role=c.message.role,
                        content=c.message.content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=c.finish_reason,
                    index=c.index,
                )
            )

        usage = None
        if raw.usage:
            usage = Usage(
                prompt_tokens=raw.usage.prompt_tokens,
                completion_tokens=raw.usage.completion_tokens,
                total_tokens=raw.usage.total_tokens,
            )

        return CompletionResponse(
            choices=choices,
            model=raw.model,
            usage=usage,
        )

    def _to_embedding_response(self, raw: Any) -> EmbeddingResponse:
        data = [
            EmbeddingData(embedding=item.embedding, index=item.index)
            for item in raw.data
        ]
        usage = None
        if raw.usage:
            usage = EmbeddingUsage(
                prompt_tokens=raw.usage.prompt_tokens,
                total_tokens=raw.usage.total_tokens,
            )
        return EmbeddingResponse(data=data, model=raw.model, usage=usage)
