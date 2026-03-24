"""Anthropic provider -- uses the ``anthropic`` SDK directly."""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportImplicitRelativeImport=false

import json
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
    Choice,
    ChoiceMessage,
    CompletionResponse,
    EmbeddingResponse,
    Usage,
)
from .base import BaseProvider

PROVIDER = "anthropic"


def _import_anthropic():
    try:
        import anthropic

        return anthropic
    except ImportError:
        raise ProviderNotAvailableError(PROVIDER, "anthropic")


class AnthropicProvider(BaseProvider):
    def __init__(self) -> None:
        anthropic = _import_anthropic()
        self._client = anthropic.AsyncAnthropic()

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **params: Any,
    ) -> CompletionResponse:
        anthropic = _import_anthropic()
        kwargs = self._build_completion_kwargs(model, messages, params)

        try:
            raw = await self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise RateLimitError(429, str(e), PROVIDER) from e
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(e.status_code, str(e), PROVIDER) from e
        except anthropic.BadRequestError as e:
            raise BadRequestError(e.status_code, str(e), PROVIDER) from e
        except anthropic.APITimeoutError as e:
            raise TimeoutError(408, str(e), PROVIDER) from e
        except anthropic.InternalServerError as e:
            raise ServerError(e.status_code, str(e), PROVIDER) from e
        except anthropic.APIStatusError as e:
            raise LLMError(e.status_code, str(e), PROVIDER) from e

        return self._to_completion_response(raw)

    async def embed(
        self,
        model: str,
        input: list[str],
        **params: Any,
    ) -> EmbeddingResponse:
        raise LLMError(
            400,
            "Anthropic does not support embeddings. "
            "Use an openai or gemini embedding model instead.",
            PROVIDER,
        )

    # -- helpers ---------------------------------------------------------------

    def _build_completion_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        system_parts, user_messages = self._split_system_messages(messages)
        converted = self._convert_messages(user_messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": converted,
            "max_tokens": params.get("max_tokens") or 4096,
        }

        if system_parts:
            kwargs["system"] = "\n".join(system_parts)
        if params.get("temperature") is not None:
            kwargs["temperature"] = params["temperature"]
        if params.get("timeout") is not None:
            kwargs["timeout"] = params["timeout"]

        tools = params.get("tools")
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        response_format = params.get("response_format")
        if (
            response_format is not None
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            schema = response_format.model_json_schema()
            kwargs["tools"] = kwargs.get("tools", []) + [
                {
                    "name": "_structured_output",
                    "description": f"Return output matching the {response_format.__name__} schema.",
                    "input_schema": schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "_structured_output"}

        return kwargs

    def _split_system_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        system: list[str] = []
        rest: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                system.append(m.get("content", ""))
            else:
                rest.append(m)
        return system, rest

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to Anthropic format."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")

            if role == "tool":
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": msg.get("content", ""),
                            }
                        ],
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                content: list[dict[str, Any]] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"]),
                        }
                    )
                result.append({"role": "assistant", "content": content})
            else:
                result.append(
                    {
                        "role": role,
                        "content": msg.get("content", "") or "",
                    }
                )
        return result

    def _convert_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert an OpenAI-format tool to Anthropic format."""
        func = tool.get("function", {})
        return {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        }

    def _to_completion_response(self, raw: Any) -> CompletionResponse:
        content_text: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in raw.content:
            if block.type == "text":
                content_text.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )

        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
        }
        finish_reason = finish_reason_map.get(raw.stop_reason, "stop")

        message = ChoiceMessage(
            role="assistant",
            content="\n".join(content_text) if content_text else None,
            tool_calls=tool_calls or None,
        )

        usage = None
        if raw.usage:
            usage = Usage(
                prompt_tokens=raw.usage.input_tokens,
                completion_tokens=raw.usage.output_tokens,
                total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
            )

        return CompletionResponse(
            choices=[Choice(message=message, finish_reason=finish_reason, index=0)],
            model=raw.model,
            usage=usage,
        )
