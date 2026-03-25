"""Google Gemini provider using the ``google-genai`` SDK.

Routing prefix: ``google/``

Authentication:
    - Env: ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``
    - Kwargs: ``api_key``

Role mapping:
    - ``system`` -> extracted to ``system_instruction`` config (accepts a list)
    - ``assistant`` -> ``model``
    - ``tool`` -> ``function_response`` part
    - ``user`` -> ``user``

Message constraints:
    - Multiple system messages: supported natively (passed as list)
    - System-only messages: raises ``BadRequestError``
    - No strict alternation required

Tool call format:
    - Tool definitions: converted to ``FunctionDeclaration``
    - Tool results: converted to ``function_response`` parts
    - Tool call IDs: synthetic (``call_0``, ``call_1``, ...) since Gemini
      doesn't provide them

Error mapping:
    - ``google.genai.errors.ClientError`` (401/403) -> ``AuthenticationError``
    - ``google.genai.errors.ClientError`` (429) -> ``RateLimitError``
    - ``google.genai.errors.ClientError`` (other) -> ``BadRequestError``
    - ``google.genai.errors.ServerError`` -> ``ServerError``
    - ``google.genai.errors.APIError`` -> ``LLMError``

Supported features:
    - Completion: yes
    - Embeddings: yes
    - Structured output (response_format): yes, via ``response_schema``

Provider-specific kwargs:
    - ``safety_settings``: override default safety settings
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

import json
import os
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
    ToolCall,
    ToolCallFunction,
    Usage,
)
from .base import BaseProvider

PROVIDER = "google"


def _import_genai():
    try:
        from google import genai

        return genai
    except ImportError:
        raise ProviderNotAvailableError(PROVIDER, "google-genai")


def _import_genai_types():
    try:
        from google.genai import types

        return types
    except ImportError:
        raise ProviderNotAvailableError(PROVIDER, "google-genai")


def _import_genai_errors():
    try:
        from google.genai import errors

        return errors
    except ImportError:
        raise ProviderNotAvailableError(PROVIDER, "google-genai")


class GoogleProvider(BaseProvider):
    def __init__(
        self,
        api_key: str | None = None,
        **_kwargs: Any,
    ) -> None:
        genai = _import_genai()
        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self._client = genai.Client(api_key=resolved_key)

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        **params: Any,
    ) -> CompletionResponse:
        genai_errors = _import_genai_errors()
        types = _import_genai_types()

        self._validate_messages(messages)
        contents = self._convert_messages(messages)
        config = self._build_config(messages, params, types)

        try:
            raw = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except genai_errors.ClientError as e:
            status = getattr(e, "code", 400)
            if status == 429:
                raise RateLimitError(429, str(e), PROVIDER) from e
            if status in (401, 403):
                raise AuthenticationError(status, str(e), PROVIDER) from e
            raise BadRequestError(status, str(e), PROVIDER) from e
        except genai_errors.ServerError as e:
            raise ServerError(getattr(e, "code", 500), str(e), PROVIDER) from e
        except genai_errors.APIError as e:
            raise LLMError(getattr(e, "code", 500), str(e), PROVIDER) from e
        except Exception as e:
            if "timed out" in str(e).lower() or "timeout" in type(e).__name__.lower():
                raise TimeoutError(408, str(e), PROVIDER) from e
            raise

        return self._to_completion_response(raw, model)

    async def embed(
        self,
        model: str,
        input: list[str],
        **params: Any,
    ) -> EmbeddingResponse:
        genai_errors = _import_genai_errors()
        types = _import_genai_types()

        config_kwargs: dict[str, Any] = {}
        if "dimensions" in params and params["dimensions"] is not None:
            config_kwargs["output_dimensionality"] = params["dimensions"]

        config = types.EmbedContentConfig(**config_kwargs) if config_kwargs else None

        try:
            raw = await self._client.aio.models.embed_content(
                model=model,
                contents=input,
                config=config,
            )
        except genai_errors.ClientError as e:
            status = getattr(e, "code", 400)
            if status == 429:
                raise RateLimitError(429, str(e), PROVIDER) from e
            if status in (401, 403):
                raise AuthenticationError(status, str(e), PROVIDER) from e
            raise BadRequestError(status, str(e), PROVIDER) from e
        except genai_errors.ServerError as e:
            raise ServerError(getattr(e, "code", 500), str(e), PROVIDER) from e
        except genai_errors.APIError as e:
            raise LLMError(getattr(e, "code", 500), str(e), PROVIDER) from e

        return self._to_embedding_response(raw, model)

    # -- validation ------------------------------------------------------------

    def _validate_messages(self, messages: list[ChatMessage]) -> None:
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

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to Gemini content format."""
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                continue
            if role == "assistant":
                role = "model"
            if role == "tool":
                parts = [
                    {
                        "function_response": {
                            "name": msg.get("tool_call_id", "unknown"),
                            "response": {"result": msg.get("content", "")},
                        }
                    }
                ]
            elif msg.get("tool_calls"):
                raw_tcs = msg["tool_calls"]
                parts = []
                for tc in raw_tcs:
                    tc_data = tc if isinstance(tc, dict) else tc.model_dump()
                    func = tc_data.get("function", tc_data)
                    parts.append(
                        {
                            "function_call": {
                                "name": func.get("name", ""),
                                "args": json.loads(func.get("arguments", "{}")),
                            }
                        }
                    )
            else:
                parts = [{"text": (msg.get("content") or "")}]
            contents.append({"role": role, "parts": parts})
        return contents

    def _build_config(
        self, messages: list[ChatMessage], params: dict[str, Any], types: Any
    ) -> Any:
        """Build a GenerateContentConfig from OpenAI-style params."""
        config_kwargs: dict[str, Any] = {}

        if params.get("temperature") is not None:
            config_kwargs["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            config_kwargs["max_output_tokens"] = params["max_tokens"]

        tools = params.get("tools")
        if tools:
            config_kwargs["tools"] = [self._convert_tool(t, types) for t in tools]

        response_format = params.get("response_format")
        if (
            response_format is not None
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_format.model_json_schema()

        system_instructions = self._extract_system_instructions(messages)
        if system_instructions:
            config_kwargs["system_instruction"] = system_instructions

        return types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    def _convert_tool(self, tool: dict[str, Any], types: Any) -> Any:
        """Convert an OpenAI-format tool to Gemini FunctionDeclaration."""
        func = tool.get("function", {})
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=func.get("name", ""),
                    description=func.get("description", ""),
                    parameters=func.get("parameters"),
                )
            ]
        )

    def _extract_system_instructions(
        self, messages: list[ChatMessage]
    ) -> list[str] | None:
        """Pull system messages out for use as system_instruction (list)."""
        parts = [
            m.get("content", "") or "" for m in messages if m.get("role") == "system"
        ]
        return parts if parts else None

    def _to_completion_response(self, raw: Any, model: str) -> CompletionResponse:
        choices: list[Choice] = []
        if not raw.candidates:
            return CompletionResponse(choices=[], model=model)

        for i, candidate in enumerate(raw.candidates):
            content = None
            tool_calls: list[ToolCall] | None = None
            finish_reason = "stop"

            if candidate.finish_reason:
                finish_reason_map = {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "content_filter",
                }
                finish_reason = finish_reason_map.get(
                    str(candidate.finish_reason), "stop"
                )

            if candidate.content and candidate.content.parts:
                text_parts = []
                fc_list: list[ToolCall] = []
                for idx, part in enumerate(candidate.content.parts):
                    if part.text is not None:
                        text_parts.append(part.text)
                    elif part.function_call is not None:
                        fc = part.function_call
                        fc_list.append(
                            ToolCall(
                                id=f"call_{idx}",
                                type="function",
                                function=ToolCallFunction(
                                    name=fc.name,
                                    arguments=json.dumps(fc.args) if fc.args else "{}",
                                ),
                            )
                        )
                content = "\n".join(text_parts) if text_parts else None
                if fc_list:
                    tool_calls = fc_list
                    finish_reason = "tool_calls"

            choices.append(
                Choice(
                    message=ChoiceMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                    index=i,
                )
            )

        usage = None
        if raw.usage_metadata:
            usage = Usage(
                prompt_tokens=raw.usage_metadata.prompt_token_count or 0,
                completion_tokens=raw.usage_metadata.candidates_token_count or 0,
                total_tokens=raw.usage_metadata.total_token_count or 0,
            )

        return CompletionResponse(choices=choices, model=model, usage=usage)

    def _to_embedding_response(self, raw: Any, model: str) -> EmbeddingResponse:
        data: list[EmbeddingData] = []
        if raw.embeddings:
            for i, emb in enumerate(raw.embeddings):
                data.append(EmbeddingData(embedding=list(emb.values), index=i))
        return EmbeddingResponse(data=data, model=model, usage=None)
