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
    - ``openai.APITimeoutError`` -> ``LLMTimeoutError``
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

import json
import logging
from collections.abc import Sequence
from typing import Any, NoReturn

from pydantic import BaseModel

from ..errors import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    LLMTimeoutError,
    ProviderNotAvailableError,
    RateLimitError,
    ServerError,
)
from ..types import (
    ChatMessage,
    Choice,
    ChoiceMessage,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    ResponseOutputFunctionCall,
    ResponseOutputText,
    ResponseResult,
    ToolCall,
    ToolCallFunction,
    ToolDef,
    Usage,
)
from ..utils import compact

logger = logging.getLogger(__name__)

PROVIDER = "openai"

KNOWN_COMPLETION_PARAMS = frozenset(
    {"temperature", "max_tokens", "timeout", "tools", "response_format", "metadata"}
)
KNOWN_EMBEDDING_PARAMS = frozenset({"dimensions"})
KNOWN_RESPONSE_PARAMS = frozenset({"temperature", "max_tokens"})


def _import_openai() -> Any:
    try:
        import openai

        return openai
    except ImportError as exc:
        raise ProviderNotAvailableError(PROVIDER, "openai") from exc


class OpenAIProvider:
    _PROVIDER = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        **_kwargs: Any,
    ) -> None:
        if _kwargs:
            logger.warning(
                "%s provider: ignoring unknown kwargs: %s", PROVIDER, sorted(_kwargs)
            )
        openai = _import_openai()
        self._client = openai.AsyncOpenAI(
            **compact(api_key=api_key, base_url=base_url, timeout=timeout)
        )

    def _map_error(self, e: Exception) -> NoReturn:
        """Map an ``openai.*`` SDK exception to the giskard error hierarchy."""
        openai = _import_openai()
        if isinstance(e, openai.RateLimitError):
            raise RateLimitError(429, str(e), self._PROVIDER) from e
        if isinstance(e, openai.AuthenticationError):
            raise AuthenticationError(e.status_code, str(e), self._PROVIDER) from e
        if isinstance(e, openai.BadRequestError):
            raise BadRequestError(e.status_code, str(e), self._PROVIDER) from e
        if isinstance(e, openai.APITimeoutError):
            raise LLMTimeoutError(408, str(e), self._PROVIDER) from e
        if isinstance(e, openai.InternalServerError):
            raise ServerError(e.status_code, str(e), self._PROVIDER) from e
        if isinstance(e, openai.APIError):
            raise LLMError(e.status_code or 500, str(e), self._PROVIDER) from e
        raise e

    async def complete(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        **params: Any,
    ) -> CompletionResponse:
        openai = _import_openai()
        self._validate_messages(messages)
        if tools is not None:
            params["tools"] = tools
        kwargs = self._build_completion_kwargs(model, messages, params)
        try:
            raw = await self._client.chat.completions.create(**kwargs)
        except openai.APIError as e:
            self._map_error(e)

        return self._to_completion_response(raw)

    async def embed(
        self,
        model: str,
        input: list[str],
        **params: Any,
    ) -> EmbeddingResponse:
        unknown = set(params) - KNOWN_EMBEDDING_PARAMS
        if unknown:
            logger.warning(
                "%s provider: ignoring unknown embedding params: %s",
                self._PROVIDER,
                sorted(unknown),
            )

        openai = _import_openai()
        kwargs: dict[str, Any] = {"model": model, "input": input}
        if (dimensions := params.get("dimensions")) is not None:
            kwargs["dimensions"] = dimensions
        try:
            raw = await self._client.embeddings.create(**kwargs)
        except openai.APIError as e:
            self._map_error(e)

        return self._to_embedding_response(raw)

    # -- validation ------------------------------------------------------------

    def _validate_messages(self, messages: Sequence[ChatMessage]) -> None:
        if not messages:
            raise BadRequestError(
                400, "Messages list must not be empty.", self._PROVIDER
            )
        has_non_system = any(m.get("role") != "system" for m in messages)
        if not has_non_system:
            raise BadRequestError(
                400,
                "Messages must contain at least one non-system message.",
                self._PROVIDER,
            )
        for m in messages:
            if m.get("role") == "tool" and not m.get("tool_call_id"):
                raise BadRequestError(
                    400, "Tool messages must have a tool_call_id.", self._PROVIDER
                )
            if m.get("role") == "system" and not (m.get("content") or "").strip():
                raise BadRequestError(
                    400, "System messages must have non-empty content.", self._PROVIDER
                )

    # -- helpers ---------------------------------------------------------------

    def _build_completion_kwargs(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Build kwargs dict for the Chat Completions API."""
        unknown = set(params) - KNOWN_COMPLETION_PARAMS
        if unknown:
            logger.warning(
                "%s provider: ignoring unknown completion params: %s",
                self._PROVIDER,
                sorted(unknown),
            )

        kwargs: dict[str, Any] = {"model": model, "messages": messages}

        if params.get("temperature") is not None:
            kwargs["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            kwargs["max_tokens"] = params["max_tokens"]
        if params.get("timeout") is not None:
            kwargs["timeout"] = params["timeout"]
        if tools := params.get("tools"):
            kwargs["tools"] = tools
        if metadata := params.get("metadata"):
            kwargs["metadata"] = metadata

        response_format = params.get("response_format")
        if response_format is not None:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                schema = response_format.model_json_schema()
                schema["additionalProperties"] = False
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                }
            kwargs["response_format"] = response_format

        return kwargs

    def _to_completion_response(self, raw: Any) -> CompletionResponse:
        """Convert raw SDK response to CompletionResponse."""
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
                            arguments=json.loads(tc.function.arguments)
                            if isinstance(tc.function.arguments, str)
                            else tc.function.arguments,
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
        """Convert raw SDK response to EmbeddingResponse."""
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

    # -- Responses API ---------------------------------------------------------

    async def respond(
        self,
        model: str,
        input: str | list[dict[str, Any]],
        *,
        instructions: str | None = None,
        previous_id: str | None = None,
        tools: list[ToolDef] | None = None,
        **params: Any,
    ) -> ResponseResult:
        unknown = set(params) - KNOWN_RESPONSE_PARAMS
        if unknown:
            logger.warning(
                "%s provider: ignoring unknown response params: %s",
                self._PROVIDER,
                sorted(unknown),
            )

        openai = _import_openai()
        kwargs: dict[str, Any] = {"model": model, "input": input}
        if instructions is not None:
            kwargs["instructions"] = instructions
        if previous_id is not None:
            kwargs["previous_response_id"] = previous_id
        if tools is not None:
            kwargs["tools"] = tools
        if params.get("temperature") is not None:
            kwargs["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            kwargs["max_output_tokens"] = params["max_tokens"]

        try:
            raw = await self._client.responses.create(**kwargs)
        except openai.APIError as e:
            self._map_error(e)

        return self._to_response_result(raw)

    def _to_response_result(self, raw: Any) -> ResponseResult:
        """Convert raw Responses API output to ResponseResult."""
        outputs: list[ResponseOutputText | ResponseOutputFunctionCall] = []
        for item in raw.output:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for content_block in getattr(item, "content", []):
                    if getattr(content_block, "type", None) == "output_text":
                        outputs.append(ResponseOutputText(text=content_block.text))
            elif item_type == "function_call":
                args = getattr(item, "arguments", "{}")
                outputs.append(
                    ResponseOutputFunctionCall(
                        call_id=getattr(item, "call_id", None),
                        name=item.name,
                        arguments=json.loads(args) if isinstance(args, str) else args,
                    )
                )

        usage = None
        if raw.usage:
            usage = Usage(
                prompt_tokens=getattr(raw.usage, "input_tokens", 0),
                completion_tokens=getattr(raw.usage, "output_tokens", 0),
                total_tokens=getattr(raw.usage, "total_tokens", 0),
            )

        return ResponseResult(
            id=raw.id,
            outputs=outputs,
            model=getattr(raw, "model", None),
            usage=usage,
        )
