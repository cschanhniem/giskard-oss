"""Google Gemini provider -- uses the ``google-genai`` SDK directly."""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

import json
import os
from typing import Any

from pydantic import BaseModel

from ..errors import (
    BadRequestError,
    LLMError,
    ProviderNotAvailableError,
    RateLimitError,
    ServerError,
)
from ..types import (
    Choice,
    ChoiceMessage,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    Usage,
)
from .base import BaseProvider

PROVIDER = "gemini"


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
    def __init__(self) -> None:
        genai = _import_genai()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=api_key)

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **params: Any,
    ) -> CompletionResponse:
        genai_errors = _import_genai_errors()
        types = _import_genai_types()

        contents = self._convert_messages(messages)
        config = self._build_config(params, types)

        try:
            raw = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except genai_errors.ClientError as e:
            if e.code == 429:
                raise RateLimitError(429, str(e), PROVIDER) from e
            raise BadRequestError(e.code, str(e), PROVIDER) from e
        except genai_errors.ServerError as e:
            raise ServerError(e.code, str(e), PROVIDER) from e
        except genai_errors.APIError as e:
            raise LLMError(e.code, str(e), PROVIDER) from e

        return self._to_completion_response(raw)

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
            if e.code == 429:
                raise RateLimitError(429, str(e), PROVIDER) from e
            raise BadRequestError(e.code, str(e), PROVIDER) from e
        except genai_errors.ServerError as e:
            raise ServerError(e.code, str(e), PROVIDER) from e
        except genai_errors.APIError as e:
            raise LLMError(e.code, str(e), PROVIDER) from e

        return self._to_embedding_response(raw)

    # -- helpers ---------------------------------------------------------------

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to Gemini content format.

        Gemini uses ``role="model"`` instead of ``role="assistant"`` and
        does not have a ``system`` role in the contents array (system
        instructions are passed separately via config).
        """
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
                parts = [
                    {
                        "function_call": {
                            "name": tc["function"]["name"],
                            "args": json.loads(tc["function"]["arguments"]),
                        }
                    }
                    for tc in msg["tool_calls"]
                ]
            else:
                parts = [{"text": msg.get("content", "") or ""}]
            contents.append({"role": role, "parts": parts})
        return contents

    def _build_config(self, params: dict[str, Any], types: Any) -> Any:
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

        system_instructions = self._extract_system_instructions(
            params.get("_original_messages", [])
        )
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
        self, messages: list[dict[str, Any]]
    ) -> str | None:
        """Pull system messages out for use as system_instruction."""
        parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        return "\n".join(parts) if parts else None

    def _to_completion_response(self, raw: Any) -> CompletionResponse:
        choices: list[Choice] = []
        if not raw.candidates:
            return CompletionResponse(choices=[], model=None)

        for i, candidate in enumerate(raw.candidates):
            content = None
            tool_calls = None
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
                fc_list = []
                for part in candidate.content.parts:
                    if part.text is not None:
                        text_parts.append(part.text)
                    elif part.function_call is not None:
                        fc = part.function_call
                        fc_list.append(
                            {
                                "id": f"call_{fc.name}",
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(fc.args)
                                    if fc.args
                                    else "{}",
                                },
                            }
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

        return CompletionResponse(choices=choices, model=None, usage=usage)

    def _to_embedding_response(self, raw: Any) -> EmbeddingResponse:
        data: list[EmbeddingData] = []
        if raw.embeddings:
            for i, emb in enumerate(raw.embeddings):
                data.append(EmbeddingData(embedding=list(emb.values), index=i))
        return EmbeddingResponse(data=data, model=None, usage=None)
