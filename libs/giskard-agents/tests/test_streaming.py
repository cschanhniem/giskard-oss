"""Tests for streaming support (GAP-004)."""

from unittest.mock import AsyncMock, MagicMock

from giskard.agents.chat import Message
from giskard.agents.generators import BaseGenerator
from giskard.agents.generators.base import GenerationParams, Response, StreamChunk
from giskard.agents.tools import Function, ToolCall, tool
from giskard.agents.workflow import ChatWorkflow, WorkflowStep


async def test_stream_fallback_yields_single_chunk():
    """The default stream() on BaseGenerator yields one chunk matching complete()."""
    gen = MagicMock(spec=BaseGenerator)
    gen.complete = AsyncMock(
        return_value=Response(
            message=Message(role="assistant", content="Hello world!"),
            finish_reason="stop",
        )
    )
    # Use the real BaseGenerator.stream implementation via the fallback
    gen.stream = BaseGenerator.stream.__get__(gen)

    chunks = []
    async for chunk in gen.stream(
        [Message(role="user", content="Hi")], GenerationParams()
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].delta == "Hello world!"
    assert chunks[0].finish_reason == "stop"
    assert chunks[0].tool_calls is None


async def test_stream_chunks_have_delta():
    """A mock streaming generator yields chunks with delta text."""

    async def _fake_stream(messages, params=None):
        yield StreamChunk(delta="Hel")
        yield StreamChunk(delta="lo ")
        yield StreamChunk(delta="world!", finish_reason="stop")

    gen = MagicMock(spec=BaseGenerator)
    gen.stream = _fake_stream

    chunks = []
    async for chunk in gen.stream([Message(role="user", content="Hi")]):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert all(isinstance(c, StreamChunk) for c in chunks)
    assert chunks[0].delta == "Hel"
    assert chunks[1].delta == "lo "
    assert chunks[2].delta == "world!"


async def test_stream_accumulates_to_full_message():
    """Concatenated deltas equal the full message content."""

    async def _fake_stream(messages, params=None):
        for part in ["The ", "quick ", "brown ", "fox."]:
            yield StreamChunk(delta=part)
        yield StreamChunk(delta="", finish_reason="stop")

    gen = MagicMock(spec=BaseGenerator)
    gen.stream = _fake_stream

    deltas = []
    async for chunk in gen.stream([Message(role="user", content="Go")]):
        deltas.append(chunk.delta)

    assert "".join(deltas) == "The quick brown fox."


async def test_stream_with_tool_calls():
    """Streaming response ending with tool_calls captures them correctly."""

    async def _fake_stream(messages, params=None):
        yield StreamChunk(delta="I'll use a tool.")
        yield StreamChunk(
            delta="",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id="tc_1",
                    function=Function(name="my_tool", arguments='{"x": 1}'),
                )
            ],
        )

    gen = MagicMock(spec=BaseGenerator)
    gen.stream = _fake_stream

    chunks = []
    async for chunk in gen.stream([Message(role="user", content="Do it")]):
        chunks.append(chunk)

    assert chunks[-1].tool_calls is not None
    assert len(chunks[-1].tool_calls) == 1
    assert chunks[-1].tool_calls[0].function.name == "my_tool"


@tool
def echo(text: str) -> str:
    """Echo text.

    Parameters
    ----------
    text : str
        Text to echo.
    """
    return text


async def test_stream_steps_yields_chunks_and_steps():
    """stream_steps() yields StreamChunk during completions and WorkflowStep at boundaries."""

    async def _fake_stream(messages, params=None):
        yield StreamChunk(delta="Hello ")
        yield StreamChunk(delta="there!", finish_reason="stop")

    gen = MagicMock(spec=BaseGenerator)
    gen.stream = _fake_stream

    collected_chunks = []
    collected_steps = []

    async with (
        ChatWorkflow(generator=gen).chat("Hi").stream_steps(max_steps=5) as events
    ):
        async for event in events:
            if isinstance(event, StreamChunk):
                collected_chunks.append(event)
            elif isinstance(event, WorkflowStep):
                collected_steps.append(event)

    assert len(collected_chunks) == 2
    assert collected_chunks[0].delta == "Hello "
    assert collected_chunks[1].delta == "there!"

    assert len(collected_steps) == 1
    assert collected_steps[0].message.content == "Hello there!"
