from collections.abc import AsyncIterator
from typing import Any, cast

from litellm import Choices, ModelResponse, acompletion
from litellm import Message as LiteLLMMessage
from litellm import _should_retry as litellm_should_retry
from pydantic import Field

from ..chat import Message
from ..tools.tool import Function, ToolCall
from .base import BaseGenerator, GenerationParams, Response, StreamChunk
from .mixins import WithRateLimiter, WithRetryPolicy


@BaseGenerator.register("litellm")
class LiteLLMGenerator(WithRateLimiter, WithRetryPolicy, BaseGenerator):
    """A generator for creating chat completion pipelines."""

    model: str = Field(
        description="The model identifier to use (e.g. 'gemini/gemini-2.0-flash')"
    )

    def _should_retry(self, err: Exception) -> bool:
        return litellm_should_retry(getattr(err, "status_code", 0))

    def _build_params(
        self, params: GenerationParams | None = None
    ) -> dict[str, Any]:
        """Merge default and override generation params into a kwargs dict.

        Parameters
        ----------
        params : GenerationParams | None
            Optional overrides for this specific call.

        Returns
        -------
        dict[str, Any]
            Keyword arguments ready to pass to litellm.
        """
        params_ = self.params.model_dump(exclude={"tools"})
        if params is not None:
            params_.update(params.model_dump(exclude={"tools"}, exclude_unset=True))

        tools = self.params.tools + (params.tools if params is not None else [])
        if tools:
            params_["tools"] = [t.to_litellm_function() for t in tools]

        return params_

    async def _complete_once(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        params_ = self._build_params(params)

        async with self._rate_limiter_context():
            response = cast(
                ModelResponse,
                await acompletion(
                    messages=[m.to_litellm() for m in messages],
                    model=self.model,
                    **params_,
                ),
            )

        choice = cast(Choices, response.choices[0])
        return Response(
            message=Message.from_litellm(cast(LiteLLMMessage, choice.message)),
            finish_reason=choice.finish_reason,  # pyright: ignore[reportArgumentType]
        )

    async def stream(
        self,
        messages: list[Message],
        params: GenerationParams | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion as incremental chunks using litellm.

        Yields ``StreamChunk`` objects as tokens arrive. Tool calls are
        accumulated across deltas and emitted as complete ``ToolCall``
        objects when the stream finishes.

        Parameters
        ----------
        messages : list[Message]
            Messages to send to the model.
        params : GenerationParams | None
            Generation parameters.

        Yields
        ------
        StreamChunk
            Incremental chunks of the response.
        """
        params_ = self._build_params(params)

        async with self._rate_limiter_context():
            response = await acompletion(
                messages=[m.to_litellm() for m in messages],
                model=self.model,
                stream=True,
                **params_,
            )

        # Accumulate tool calls by index since they arrive as deltas
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        async for chunk in response:
            choice = chunk.choices[0]
            delta = choice.delta

            # Merge incremental tool_call deltas
            if delta_tcs := getattr(delta, "tool_calls", None):
                for dtc in delta_tcs:
                    idx = dtc.index if hasattr(dtc, "index") else 0
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": getattr(dtc, "id", None) or "",
                            "name": "",
                            "arguments": "",
                        }
                    entry = tool_calls_by_index[idx]
                    if dtc_id := getattr(dtc, "id", None):
                        entry["id"] = dtc_id
                    if fn := getattr(dtc, "function", None):
                        if fn_name := getattr(fn, "name", None):
                            entry["name"] += fn_name
                        if fn_args := getattr(fn, "arguments", None):
                            entry["arguments"] += fn_args

            finish_reason = getattr(choice, "finish_reason", None)
            text_delta = getattr(delta, "content", None) or ""

            # On the final chunk, emit accumulated tool calls
            resolved_tool_calls: list[ToolCall] | None = None
            if finish_reason and tool_calls_by_index:
                resolved_tool_calls = [
                    ToolCall(
                        id=entry["id"],
                        function=Function(
                            name=entry["name"],
                            arguments=entry["arguments"],
                        ),
                    )
                    for _, entry in sorted(tool_calls_by_index.items())
                ]

            yield StreamChunk(
                delta=text_delta,
                finish_reason=finish_reason,
                tool_calls=resolved_tool_calls,
            )
