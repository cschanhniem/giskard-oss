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
    - ``anthropic.APITimeoutError`` -> ``LLMTimeoutError``
    - ``anthropic.InternalServerError`` -> ``ServerError``
    - ``anthropic.APIStatusError`` -> ``LLMError``

Supported features:
    - Completion: yes
    - Embeddings: no (provider does not implement ``EmbeddingProvider``)
    - Structured output (response_format): yes, via native ``output_config`` (json_schema)

Provider-specific kwargs (configure-time):
    - ``merge_system``: if True, concatenate multiple system messages instead of raising
    - ``base_url``: custom API endpoint
    - ``timeout``: request timeout in seconds
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportImplicitRelativeImport=false

import json
import logging
from collections.abc import Sequence
from typing import Any, Literal, NoReturn, TypedDict

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
    AssistantMessage,
    ChatCompletion,
    Choice,
    Function,
    FunctionCall,
    FunctionToolDefinition,
    Message,
    SystemMessage,
    ToolCall,
    ToolInput,
    ToolMessage,
    Usage,
    validate_messages,
    validate_tools,
)
from ..utils import compact

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"

KNOWN_COMPLETION_PARAMS = frozenset(
    {"temperature", "max_tokens", "timeout", "tools", "response_format"}
)


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


def _import_anthropic() -> Any:
    try:
        import anthropic

        return anthropic
    except ImportError as exc:
        raise ProviderNotAvailableError(PROVIDER, "anthropic") from exc


class AnthropicProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        merge_system: bool = False,
        **_kwargs: Any,
    ) -> None:
        if _kwargs:
            logger.warning(
                "%s provider: ignoring unknown kwargs: %s", PROVIDER, sorted(_kwargs)
            )
        anthropic = _import_anthropic()
        self._merge_system = merge_system
        self._client = anthropic.AsyncAnthropic(
            **compact(api_key=api_key, base_url=base_url, timeout=timeout)
        )

    def _map_error(self, e: Exception) -> NoReturn:
        """Map an ``anthropic.*`` SDK exception to the giskard error hierarchy."""
        anthropic = _import_anthropic()
        if isinstance(e, anthropic.RateLimitError):
            raise RateLimitError(429, str(e), PROVIDER) from e
        if isinstance(e, anthropic.AuthenticationError):
            raise AuthenticationError(e.status_code, str(e), PROVIDER) from e
        if isinstance(e, anthropic.BadRequestError):
            raise BadRequestError(e.status_code, str(e), PROVIDER) from e
        if isinstance(e, anthropic.APITimeoutError):
            raise LLMTimeoutError(408, str(e), PROVIDER) from e
        if isinstance(e, anthropic.InternalServerError):
            raise ServerError(e.status_code, str(e), PROVIDER) from e
        if isinstance(e, anthropic.APIStatusError):
            raise LLMError(e.status_code, str(e), PROVIDER) from e
        if isinstance(e, anthropic.APIError):
            raise LLMError(
                getattr(e, "status_code", None) or 500, str(e), PROVIDER
            ) from e
        raise e

    async def complete(
        self,
        model: str,
        messages: Sequence[Message | dict[str, Any]],
        *,
        tools: Sequence[ToolInput] | None = None,
        **params: Any,
    ) -> ChatCompletion:
        anthropic = _import_anthropic()
        messages = validate_messages(*messages)
        self._validate_messages(messages)
        if tools is not None:
            params["tools"] = validate_tools(*tools)
        kwargs = self._build_completion_kwargs(model, messages, params)

        try:
            raw = await self._client.messages.create(**kwargs)
        except anthropic.APIError as e:
            self._map_error(e)

        return self._to_completion_response(raw)

    # -- validation ------------------------------------------------------------

    def _validate_messages(self, messages: Sequence[Message]) -> None:
        if not messages:
            raise BadRequestError(400, "Messages list must not be empty.", PROVIDER)

        system_count = sum(
            1 for message in messages if isinstance(message, SystemMessage)
        )
        has_non_system = any(
            not isinstance(message, SystemMessage) for message in messages
        )
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

        non_system = [
            message for message in messages if not isinstance(message, SystemMessage)
        ]
        for i in range(1, len(non_system)):
            prev_role = non_system[i - 1].role
            curr_role = non_system[i].role
            # Skip: consecutive tool messages are valid (they merge into
            # a single user message with multiple tool_result blocks).
            if prev_role == "tool" or curr_role == "tool":
                continue
            if prev_role == curr_role:
                raise BadRequestError(
                    400,
                    f"Anthropic requires alternating user/assistant messages, "
                    f"but found consecutive '{curr_role}' messages.",
                    PROVIDER,
                )

        for message in messages:
            if isinstance(message, ToolMessage) and not message.tool_call_id:
                raise BadRequestError(
                    400, "Tool messages must have a tool_call_id.", PROVIDER
                )
            if isinstance(message, SystemMessage) and (
                message.text is None or not message.text.strip()
            ):
                raise BadRequestError(
                    400, "System messages must have non-empty content.", PROVIDER
                )

    # -- helpers ---------------------------------------------------------------

    def _build_completion_kwargs(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Build kwargs dict for the Anthropic Messages API."""
        unknown = set(params) - KNOWN_COMPLETION_PARAMS
        if unknown:
            logger.warning(
                "%s provider: ignoring unknown completion params: %s",
                PROVIDER,
                sorted(unknown),
            )

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

        if tools := params.get("tools"):
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        response_format = params.get("response_format")
        if (
            response_format is not None
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            schema = response_format.model_json_schema()
            schema["additionalProperties"] = False
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            }

        return kwargs

    def _split_system_messages(
        self, messages: Sequence[Message | dict[str, Any]]
    ) -> tuple[list[str], list[Message]]:
        """Separate system messages from conversation messages."""
        validated_messages = validate_messages(*messages)
        system: list[str] = []
        rest: list[Message] = []
        for message in validated_messages:
            if isinstance(message, SystemMessage):
                system.append(message.text or "")
            else:
                rest.append(message)
        return system, rest

    def _convert_messages(
        self, messages: Sequence[Message | dict[str, Any]]
    ) -> list[_AnthropicMessage]:
        """Convert OpenAI-format messages to Anthropic format."""
        validated_messages = validate_messages(*messages)
        result: list[_AnthropicMessage] = []
        for message in validated_messages:
            if isinstance(message, ToolMessage):
                block: _ToolResultContent = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.text or "",
                }
                # Merge consecutive tool messages into a single user message
                prev = result[-1] if result else None
                if (
                    prev
                    and prev["role"] == "user"
                    and isinstance(prev["content"], list)
                ):
                    prev["content"].append(block)  # type: ignore[union-attr]
                else:
                    result.append({"role": "user", "content": [block]})
            elif isinstance(message, AssistantMessage) and message.tool_calls:
                content: list[_ToolResultContent | _ToolUseContent | _TextContent] = []
                if message.text:
                    content.append({"type": "text", "text": message.text})
                for tool_call in message.tool_calls:
                    tool_call_data = tool_call.model_dump()
                    func = tool_call_data.get("function", tool_call_data)
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call_data.get("id", ""),
                            "name": func.get("name", ""),
                            "input": json.loads(func.get("arguments", "{}")),
                        }
                    )
                result.append({"role": "assistant", "content": content})
            else:
                result.append(
                    {
                        "role": message.role,
                        "content": getattr(message, "text", None) or "",
                    }
                )
        return result

    def _convert_tool(self, tool: FunctionToolDefinition) -> dict[str, Any]:
        """Convert an OpenAI-format tool to Anthropic format."""
        return {
            "name": tool.function.name,
            "description": tool.function.description,
            "input_schema": tool.function.parameters,
        }

    def _to_completion_response(self, raw: Any) -> ChatCompletion:
        """Convert raw SDK response to ChatCompletion."""
        content_text: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in raw.content:
            if block.type == "text":
                content_text.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    FunctionCall(
                        id=block.id,
                        function=Function(
                            name=block.name,
                            arguments=json.dumps(block.input)
                            if isinstance(block.input, dict)
                            else block.input,
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

        content: str | None = "\n".join(content_text) if content_text else None
        message = AssistantMessage(
            content=content,
            tool_calls=tool_calls or None,
        )

        usage = None
        if raw.usage:
            usage = Usage(
                prompt_tokens=raw.usage.input_tokens,
                completion_tokens=raw.usage.output_tokens,
                total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
            )

        return ChatCompletion(
            choices=[Choice(message=message, finish_reason=finish_reason, index=0)],
            model=raw.model,
            usage=usage,
        )
