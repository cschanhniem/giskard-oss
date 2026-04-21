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

import logging
from collections.abc import Sequence
from typing import Any, NoReturn

from pydantic import ValidationError

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
    CompletionResponse,
    EasyInputMessage,
    EmbeddingResponse,
    FunctionTool,
    ResponseResult,
)
from ..types._openai_chat import ChatParameters
from ..types._openai_embedding import EmbeddingParameters
from ..types._openai_reponse import ResponseParameters
from ..utils import compact

logger = logging.getLogger(__name__)

PROVIDER = "openai"

KNOWN_EMBEDDING_PARAMS = frozenset({"dimensions"})


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
            raise LLMError(
                getattr(e, "status_code", None) or 500, str(e), self._PROVIDER
            ) from e
        raise e

    async def complete(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        tools: list[FunctionTool] | None = None,
        **params: Any,
    ) -> CompletionResponse:
        openai = _import_openai()

        try:
            chat_parameters = ChatParameters.model_validate(
                {"model": model, "messages": messages, "tools": tools, **params}
            )
        except ValidationError as e:
            raise BadRequestError(400, str(e), self._PROVIDER) from e

        kwargs = chat_parameters.model_dump(exclude_none=True)
        dropped_keys = set(params) - set(kwargs)
        if dropped_keys:
            logger.warning(
                "%s provider: ignoring unknown completion params: %s",
                self._PROVIDER,
                sorted(dropped_keys),
            )

        try:
            raw = await self._client.chat.completions.create(**kwargs)
        except openai.APIError as e:
            self._map_error(e)

        return CompletionResponse.model_validate(raw.model_dump())

    async def embed(
        self,
        model: str,
        input: list[str],
        **params: Any,
    ) -> EmbeddingResponse:
        try:
            embedding_parameters = EmbeddingParameters.model_validate(
                {"model": model, "input": input, **params}
            )
        except ValidationError as e:
            raise BadRequestError(400, str(e), self._PROVIDER) from e

        kwargs = embedding_parameters.model_dump(exclude_none=True)
        dropped_keys = set(params) - set(kwargs)
        if dropped_keys:
            logger.warning(
                "%s provider: ignoring unknown embedding params: %s",
                self._PROVIDER,
                sorted(dropped_keys),
            )

        openai = _import_openai()
        try:
            raw = await self._client.embeddings.create(**kwargs)
        except openai.APIError as e:
            self._map_error(e)

        return EmbeddingResponse.model_validate(raw.model_dump())

    # -- Responses API ---------------------------------------------------------

    async def respond(
        self,
        model: str,
        input: str | list[EasyInputMessage],
        *,
        instructions: str | None = None,
        previous_id: str | None = None,
        tools: list[FunctionTool] | None = None,
        **params: Any,
    ) -> ResponseResult:
        openai = _import_openai()

        try:
            response_parameters = ResponseParameters.model_validate(
                {
                    "model": model,
                    "input": input,
                    "instructions": instructions,
                    "previous_response_id": previous_id,
                    "tools": tools,
                    **params,
                }
            )
        except ValidationError as e:
            raise BadRequestError(400, str(e), self._PROVIDER) from e

        kwargs = response_parameters.model_dump(exclude_none=True)
        dropped_keys = set(params) - set(kwargs)
        if dropped_keys:
            logger.warning(
                "%s provider: ignoring unknown response params: %s",
                self._PROVIDER,
                sorted(dropped_keys),
            )

        try:
            raw = await self._client.responses.create(**kwargs)
        except openai.APIError as e:
            self._map_error(e)

        return ResponseResult.model_validate(raw.model_dump())
