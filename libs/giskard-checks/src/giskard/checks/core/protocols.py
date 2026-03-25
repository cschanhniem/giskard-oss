from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderableType, RenderResult


class InteractionGenerator[YieldType, SendType](Protocol):
    def generate(self, trace: SendType) -> AsyncGenerator[YieldType, SendType]: ...


@runtime_checkable
class RichConsoleProtocol(Protocol):
    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult": ...


@runtime_checkable
class RichProtocol(Protocol):
    def __rich__(self) -> "RenderableType": ...
