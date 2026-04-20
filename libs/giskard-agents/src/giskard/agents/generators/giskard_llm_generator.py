from typing import Any, cast, override

from giskard import llm
from giskard.llm import acompletion, should_retry
from pydantic import Field

from ..chat import Message
from ..tools import Function, Tool, ToolCall
from ._types import FinishReason, GenerationParams, Response
from .base import BaseGenerator
from .middleware import CompletionMiddleware, RetryMiddleware, RetryPolicy


@CompletionMiddleware.register("giskard_llm_retry")
class GiskardLLMRetryMiddleware(RetryMiddleware):
    """Retry middleware that checks error types for retry eligibility."""

    @override
    def _should_retry(self, err: Exception) -> bool:
        return should_retry(err)


@BaseGenerator.register("giskard_llm")
class GiskardLLMGenerator(BaseGenerator):
    """A generator for creating chat completion pipelines."""

    model: str = Field(
        description="The model identifier to use (e.g. 'google/gemini-2.0-flash')"
    )
    retry_policy: RetryPolicy | None = Field(default_factory=RetryPolicy)

    @override
    def _create_retry_middleware(self) -> GiskardLLMRetryMiddleware | None:
        if self.retry_policy is None:
            return None
        return GiskardLLMRetryMiddleware(retry_policy=self.retry_policy)

    def _serialize_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Convert ``Tool`` objects to the OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                },
            }
            for t in tools
        ]

    def _serialize_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal ``Message`` objects to the wire dict format accepted by giskard-llm."""
        return [
            m.model_dump(include={"role", "content", "tool_calls", "tool_call_id"})
            for m in messages
        ]

    def _deserialize_response(self, message: llm.AssistantMessage) -> Message:
        """Convert a giskard-llm ``AssistantMessage`` into an internal ``Message``."""
        tool_calls: list[ToolCall] | None = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    function=Function(
                        name=tc.function.name, arguments=tc.function.arguments
                    ),
                )
                for tc in message.tool_calls
                if isinstance(tc, llm.FunctionCall)
            ] or None

        return Message(
            role="assistant",
            content=message.text,
            tool_calls=tool_calls,
        )

    @override
    async def _call_model(
        self,
        messages: list[Message],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> Response:
        wire_messages = self._serialize_messages(messages)
        wire_params = params.model_dump(exclude={"tools"})
        wire_tools = self._serialize_tools(params.tools) if params.tools else []
        if wire_tools:
            wire_params["tools"] = wire_tools
        if metadata:
            wire_params["metadata"] = metadata

        raw: llm.ChatCompletion = await acompletion(
            messages=wire_messages, model=self.model, **wire_params
        )

        choice = raw.choices[0]
        message = self._deserialize_response(choice.message)
        response_metadata = raw.model_dump(exclude={"choices"})
        return Response(
            message=message,
            finish_reason=cast(FinishReason, choice.finish_reason),
            metadata=response_metadata,
        )
