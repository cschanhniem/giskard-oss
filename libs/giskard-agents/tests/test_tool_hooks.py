"""Tests for ToolHook lifecycle hooks (GAP-001)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from giskard.agents.chat import Message
from giskard.agents.generators import BaseGenerator
from giskard.agents.generators.base import Response
from giskard.agents.tools import Function, Tool, ToolCall, tool
from giskard.agents.workflow import ChatWorkflow, ToolHook


@tool
def greet(name: str) -> str:
    """Greet someone.

    Parameters
    ----------
    name : str
        Name to greet.
    """
    return f"Hello, {name}!"


def _mock_generator_with_tool_call(tool_name: str, args: str) -> MagicMock:
    """Return a mock generator that first issues a tool call, then stops."""
    gen = MagicMock(spec=BaseGenerator)
    gen.complete = AsyncMock(
        side_effect=[
            Response(
                message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="tc_1",
                            function=Function(name=tool_name, arguments=args),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            ),
            Response(
                message=Message(role="assistant", content="Done."),
                finish_reason="stop",
            ),
        ]
    )
    return gen


async def test_before_hook_called_before_execution():
    """before_tool_call receives the correct tool, tool_call, and arguments."""
    received = {}

    class Spy(ToolHook):
        async def before_tool_call(
            self, tool: Tool, tool_call: ToolCall, arguments: dict[str, Any]
        ) -> bool | None:
            received["tool_name"] = tool.name
            received["tc_id"] = tool_call.id
            received["args"] = arguments
            return None

    gen = _mock_generator_with_tool_call("greet", '{"name": "Alice"}')
    chat = await (
        ChatWorkflow(generator=gen)
        .chat("Say hi to Alice")
        .with_tools(greet)
        .with_tool_hooks(Spy())
        .run(max_steps=3)
    )

    assert received["tool_name"] == "greet"
    assert received["tc_id"] == "tc_1"
    assert received["args"] == {"name": "Alice"}
    assert not chat.failed


async def test_before_hook_blocks_execution():
    """Returning False from before_tool_call blocks the tool and returns an error message."""
    fn_called = False
    original_fn = greet.fn

    @tool
    def tracked_greet(name: str) -> str:
        """Greet someone.

        Parameters
        ----------
        name : str
            Name.
        """
        nonlocal fn_called
        fn_called = True
        return original_fn(name)

    class Blocker(ToolHook):
        async def before_tool_call(
            self, tool: Tool, tool_call: ToolCall, arguments: dict[str, Any]
        ) -> bool | None:
            return False

    gen = _mock_generator_with_tool_call("tracked_greet", '{"name": "Alice"}')
    chat = await (
        ChatWorkflow(generator=gen)
        .chat("Say hi")
        .with_tools(tracked_greet)
        .with_tool_hooks(Blocker())
        .run(max_steps=3)
    )

    assert not fn_called
    tool_msgs = [m for m in chat.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert isinstance(tool_msgs[0].content, str)
    assert "blocked" in tool_msgs[0].content.lower()


async def test_after_hook_receives_result():
    """after_tool_call is called with the tool result."""
    received = {}

    class Spy(ToolHook):
        async def after_tool_call(
            self, tool: Tool, tool_call: ToolCall, result: Any
        ) -> None:
            received["result"] = result

    gen = _mock_generator_with_tool_call("greet", '{"name": "Bob"}')
    await (
        ChatWorkflow(generator=gen)
        .chat("Say hi to Bob")
        .with_tools(greet)
        .with_tool_hooks(Spy())
        .run(max_steps=3)
    )

    assert received["result"] == "Hello, Bob!"


async def test_multiple_hooks_all_called():
    """All hooks in the list are called."""
    calls = []

    class HookA(ToolHook):
        async def before_tool_call(
            self, tool: Tool, tool_call: ToolCall, arguments: dict[str, Any]
        ) -> bool | None:
            calls.append("A_before")
            return None

        async def after_tool_call(
            self, tool: Tool, tool_call: ToolCall, result: Any
        ) -> None:
            calls.append("A_after")

    class HookB(ToolHook):
        async def before_tool_call(
            self, tool: Tool, tool_call: ToolCall, arguments: dict[str, Any]
        ) -> bool | None:
            calls.append("B_before")
            return None

        async def after_tool_call(
            self, tool: Tool, tool_call: ToolCall, result: Any
        ) -> None:
            calls.append("B_after")

    gen = _mock_generator_with_tool_call("greet", '{"name": "X"}')
    await (
        ChatWorkflow(generator=gen)
        .chat("Hi")
        .with_tools(greet)
        .with_tool_hooks(HookA(), HookB())
        .run(max_steps=3)
    )

    assert calls == ["A_before", "B_before", "A_after", "B_after"]


async def test_no_hooks_default_behavior():
    """Workflow without hooks works identically to before."""
    gen = _mock_generator_with_tool_call("greet", '{"name": "Default"}')
    chat = await (
        ChatWorkflow(generator=gen).chat("Hi").with_tools(greet).run(max_steps=3)
    )

    assert not chat.failed
    assert chat.last.content == "Done."
