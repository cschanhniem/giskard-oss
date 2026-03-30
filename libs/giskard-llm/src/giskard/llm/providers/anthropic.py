"""Anthropic provider using the ``anthropic`` SDK.

Routing prefix: ``anthropic/``

Authentication:
    - Env: ``ANTHROPIC_API_KEY`` (read by the SDK automatically)
    - Kwargs: ``api_key``, ``base_url``, ``timeout``

Role mapping:
    - ``system`` -> extracted to top-level ``system`` param (single string)
    - ``user`` -> ``user``
    - ``assistant`` -> ``assistant``
    - ``tool`` -> wrapped as ``user`` with ``tool_result`` content block

Message constraints:
    - Multiple system messages: raises ``BadRequestError`` by default.
      Configure with ``merge_system=True`` to concatenate them.
    - Consecutive same-role messages: raises ``BadRequestError``
      (strict alternation required by the Anthropic API).
    - System-only messages: raises ``BadRequestError``

Tool call format:
    - Tool definitions: converted to Anthropic ``{name, description, input_schema}``
    - Tool results: converted to ``tool_result`` content blocks in ``user`` messages
    - Tool call IDs: preserved from Anthropic's ``tool_use`` blocks

Error mapping:
    - ``anthropic.RateLimitError`` -> ``RateLimitError``
    - ``anthropic.AuthenticationError`` -> ``AuthenticationError``
    - ``anthropic.BadRequestError`` -> ``BadRequestError``
    - ``anthropic.APITimeoutError`` -> ``TimeoutError``
    - ``anthropic.InternalServerError`` -> ``ServerError``
    - ``anthropic.APIStatusError`` -> ``LLMError``

Supported features:
    - Completion: yes
    - Embeddings: no (raises ``LLMError``)
    - Structured output (response_format): yes, via forced tool call

Provider-specific kwargs (configure-time):
    - ``merge_system``: if True, concatenate multiple system messages instead of raising
    - ``base_url``: custom API endpoint
    - ``timeout``: request timeout in seconds
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportImplicitRelativeImport=false

import json
from collections.abc import Sequence
from typing import Any, Literal, TypedDict

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
    EmbeddingResponse,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from .base import BaseProvider

PROVIDER = "anthropic"


# -- Private wire-format TypedDicts -------------------------------------------


class _ToolResultContent(TypedDict):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str


class _ToolUseContent(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class _TextContent(TypedDict):
    type: Literal["text"]
    text: str


class _AnthropicMessage(TypedDict):
    role: str
    content: str | list[_ToolResultContent | _ToolUseContent | _TextContent]


def _import_anthropic():
    try:
        import anthropic

        return anthropic
    except ImportError:
        raise ProviderNotAvailableError(PROVIDER, "anthropic")


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        merge_system: bool = False,
        **_kwargs: Any,
    ) -> None:
        anthropic = _import_anthropic()
        self._merge_system = merge_system
        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client = anthropic.AsyncAnthropic(**client_kwargs)

    async def complete(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        **params: Any,
    ) -> CompletionResponse:
        anthropic = _import_anthropic()
        self._validate_messages(messages)
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
            "Use an openai or google embedding model instead.",
            PROVIDER,
        )

    # -- validation ------------------------------------------------------------

    def _validate_messages(self, messages: Sequence[ChatMessage]) -> None:
        if not messages:
            raise BadRequestError(400, "Messages list must not be empty.", PROVIDER)

        system_count = sum(1 for m in messages if m.get("role") == "system")
        has_non_system = any(m.get("role") != "system" for m in messages)
        if not has_non_system:
            raise BadRequestError(
                400, "Messages must contain at least one non-system message.", PROVIDER
            )

        if system_count > 1 and not self._merge_system:
            raise BadRequestError(
                400,
                "Anthropic does not support multiple system messages. "
                "Configure with merge_system=True to concatenate them.",
                PROVIDER,
            )

        non_system = [m for m in messages if m.get("role") != "system"]
        for i in range(1, len(non_system)):
            prev_role = non_system[i - 1].get("role")
            curr_role = non_system[i].get("role")
            if prev_role == curr_role:
                raise BadRequestError(
                    400,
                    f"Anthropic requires alternating user/assistant messages, "
                    f"but found consecutive '{curr_role}' messages.",
                    PROVIDER,
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
        self, messages: Sequence[ChatMessage]
    ) -> tuple[list[str], list[ChatMessage]]:
        system: list[str] = []
        rest: list[ChatMessage] = []
        for m in messages:
            if m.get("role") == "system":
                system.append(m.get("content", "") or "")
            else:
                rest.append(m)
        return system, rest

    def _convert_messages(
        self, messages: Sequence[ChatMessage]
    ) -> list[_AnthropicMessage]:
        """Convert OpenAI-format messages to Anthropic format."""
        result: list[_AnthropicMessage] = []
        for msg in messages:
            role = msg.get("role", "user")

            if role == "tool":
                block: _ToolResultContent = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", "") or "",
                }
                result.append({"role": "user", "content": [block]})
            elif role == "assistant" and msg.get("tool_calls"):
                content: list[_ToolUseContent | _TextContent] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg.get("content") or ""})
                for tc in msg.get("tool_calls", []):
                    tc_func = tc if isinstance(tc, dict) else tc.model_dump()
                    func = tc_func.get("function", tc_func)
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc_func.get("id", ""),
                            "name": func.get("name", ""),
                            "input": json.loads(func.get("arguments", "{}")),
                        }
                    )
                result.append({"role": "assistant", "content": content})  # pyright: ignore[reportArgumentType]
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
        tool_calls: list[ToolCall] = []

        for block in raw.content:
            if block.type == "text":
                content_text.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=ToolCallFunction(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
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
