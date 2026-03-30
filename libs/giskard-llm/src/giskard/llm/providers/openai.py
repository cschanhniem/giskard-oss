"""OpenAI provider using the ``openai`` SDK.

Routing prefix: ``openai/`` (also the default when no prefix is given)

Authentication:
    - Env: ``OPENAI_API_KEY`` (read by the SDK automatically)
    - Kwargs: ``api_key``, ``base_url``, ``timeout``

Role mapping:
    All canonical roles (system, user, assistant, tool) are passed through
    as-is — OpenAI supports them natively.

Message constraints:
    - Multiple system messages: supported natively
    - System-only messages: raises ``BadRequestError``
    - No strict alternation required

Tool call format:
    Tool definitions and results use the OpenAI format natively.

Error mapping:
    - ``openai.RateLimitError`` -> ``RateLimitError``
    - ``openai.AuthenticationError`` -> ``AuthenticationError``
    - ``openai.BadRequestError`` -> ``BadRequestError``
    - ``openai.APITimeoutError`` -> ``TimeoutError``
    - ``openai.InternalServerError`` -> ``ServerError``
    - ``openai.APIError`` -> ``LLMError``

Supported features:
    - Completion: yes
    - Embeddings: yes
    - Structured output (response_format): yes (passed through to SDK)

Provider-specific kwargs:
    - ``base_url``: custom API endpoint
    - ``timeout``: request timeout in seconds
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportImplicitRelativeImport=false

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

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
    ChatMessage,
    Choice,
    ChoiceMessage,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    ToolCall,
    ToolCallFunction,
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
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        **_kwargs: Any,
    ) -> None:
        openai = _import_openai()
        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client = openai.AsyncOpenAI(**client_kwargs)

    async def complete(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        **params: Any,
    ) -> CompletionResponse:
        openai = _import_openai()
        self._validate_messages(messages)
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

    # -- validation ------------------------------------------------------------

    def _validate_messages(self, messages: Sequence[ChatMessage]) -> None:
        if not messages:
            raise BadRequestError(400, "Messages list must not be empty.", PROVIDER)
        has_non_system = any(m.get("role") != "system" for m in messages)
        if not has_non_system:
            raise BadRequestError(
                400, "Messages must contain at least one non-system message.", PROVIDER
            )
        for m in messages:
            if m.get("role") == "tool" and not m.get("tool_call_id"):
                raise BadRequestError(
                    400, "Tool messages must have a tool_call_id.", PROVIDER
                )
            if m.get("role") == "system" and not (m.get("content") or "").strip():
                raise BadRequestError(
                    400, "System messages must have non-empty content.", PROVIDER
                )

    # -- helpers ---------------------------------------------------------------

    def _build_completion_kwargs(
        self,
        model: str,
        messages: Sequence[ChatMessage],
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
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": response_format.model_json_schema(),
                    },
                }
            kwargs["response_format"] = response_format

        return kwargs

    def _to_completion_response(self, raw: Any) -> CompletionResponse:
        choices = []
        for c in raw.choices:
            tool_calls = None
            if c.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function=ToolCallFunction(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
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
