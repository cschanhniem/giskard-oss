from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, cast, override

from giskard.core import Discriminated, discriminated_base
from pydantic import Field, PrivateAttr, model_validator
from rich.console import Console, ConsoleOptions, RenderResult

from ..utils.parameter_injection import ParameterInjectionRequirement
from ..utils.value_provider import (
    StaticValueProvider,
    ValueGeneratorProvider,
    ValueProvider,
)
from .input_generator import InputGenerator
from .trace import Trace
from .types import GeneratorType, ProviderType

INJECTABLE_TRACE = ParameterInjectionRequirement(
    class_info=Trace,
    optional=True,
)

INJECTABLE_INPUT = ParameterInjectionRequirement(
    class_info=Any,
    optional=True,
)


@discriminated_base
class BaseInteraction[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Discriminated
):
    """Base class for interaction specifications that generate interactions.

    An interaction spec produces one or more `Interaction` objects by yielding
    them through an async generator. Each yielded interaction receives the updated
    trace (including the newly yielded interaction) via `generator.asend()`.

    This allows for multi-turn interactions where subsequent inputs can depend
    on the accumulated trace history.

    Subclasses must implement `generate()` to produce interactions. They should
    be registered using `@BaseInteraction.register("kind")` for polymorphic
    serialization.

    Attributes
    ----------
    InputType : TypeVar
        Type of the input values for interactions
    OutputType : TypeVar
        Type of the output values for interactions
    """

    def generate(
        self, trace: TraceType
    ) -> AsyncGenerator[Interaction[InputType, OutputType, TraceType], TraceType]:
        """Generate interactions from the current trace state.

        This method is called by the scenario runner to produce interactions.
        It yields `Interaction` objects and receives updated traces (including
        the newly yielded interaction) via the async generator protocol.

        Parameters
        ----------
        trace : TraceType
            The current trace state before generating the interaction.

        Yields
        ------
        Interaction[InputType, OutputType]
            An interaction to add to the trace.

        Receives
        --------
        TraceType
            The updated trace after the yielded interaction was added.
            Use `generator.asend(updated_trace)` to receive this value.

        Examples
        --------
        ```python
        async def generate(self, trace: TraceType) -> AsyncGenerator[Interaction, TraceType]:
            # Generate first interaction
            interaction = Interaction(inputs="hello", outputs="hi")
            updated_trace = yield interaction

            # Generate second interaction based on updated trace
            next_input = f"Previous had {len(updated_trace.interactions)} interactions"
            interaction = Interaction(inputs=next_input, outputs="response")
            yield interaction
        ```
        """
        raise NotImplementedError


@BaseInteraction.register("interaction")
class Interaction[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseInteraction[InputType, OutputType, TraceType]
):
    """Unified interaction class supporting static values, callables, and generators.

    **Note**: For most use cases, the fluent API (`scenario().interact()`) is recommended
    as it automatically creates `Interaction` objects and is simpler to use. This class
    is useful for advanced use cases where you need direct control over interaction specification.

    This class serves dual roles:
    - **Spec** (dynamic): When `inputs`/`outputs` are callables, `_resolved=False`. The
      interaction generates resolved interactions via `generate()`.
    - **Record** (resolved): When `inputs`/`outputs` are static values, `_resolved=True`.
      The interaction yields itself directly and can be stored in a `Trace`.

    The `inputs` field can be:
    - A static value of type `InputType`
    - A callable with no arguments that returns `InputType` (or awaitable/generator)
    - A callable that takes the current `Trace` and returns `InputType` (or awaitable/generator)
    - A generator/async generator that yields `InputType` values

    The `outputs` field can be:
    - A static value of type `OutputType`
    - A callable that takes `InputType` and returns `OutputType` (or awaitable)
    - A callable that takes `(InputType, Trace)` and returns `OutputType` (or awaitable)
    - A callable that returns an `Interaction` object directly

    When using generators for inputs, the spec will yield multiple interactions,
    one for each input value produced by the generator. Each interaction receives
    the updated trace (including previous interactions) via the generator protocol.

    Attributes
    ----------
    inputs : InputType | Callable[..., InputType | Awaitable[InputType] | Generator | AsyncGenerator]
        Input specification. Can be a static value, callable, or generator.
        Callables can take no arguments or the current `Trace` as an argument.
        Generators yield multiple inputs and receive updated traces via `asend()`.
    outputs : OutputType | Callable[..., OutputType | Awaitable[OutputType | Interaction]]
        Output specification. Can be a static value or callable.
        Callables receive the current `InputType` and optionally the current `Trace`.
        Can return an `Interaction` object directly to override default metadata.
    metadata : dict[str, Any]
        Default metadata to attach to interactions. Can be overridden if `outputs`
        returns an `Interaction` object directly.

    Examples
    --------
    Static inputs and outputs:
    ```python
    Interaction(
        inputs="Hello",
        outputs="Hi there!",
        metadata={"source": "test"}
    )
    ```

    Callable-based outputs:
    ```python
    Interaction(
        inputs="What is 2+2?",
        outputs=lambda inputs: f"Answer: {eval(inputs)}"
    )
    ```

    Trace-dependent inputs:
    ```python
    Interaction(
        inputs=lambda trace: f"Message #{len(trace.interactions) + 1}",
        outputs=lambda inputs, trace: f"Received: {inputs}"
    )
    ```

    Generator for multiple interactions:
    ```python
    async def input_gen(trace: Trace) -> AsyncGenerator[str, Trace]:
        for i in range(3):
            yield f"Message {i+1}"

    Interaction(
        inputs=input_gen,
        outputs=lambda inputs: f"Echo: {inputs}"
    )
    ```
    """

    inputs: (
        InputGenerator[InputType, TraceType]
        | GeneratorType[[], InputType, None]
        | GeneratorType[[TraceType], InputType, TraceType]
    ) = Field(..., description="The inputs of the interaction.")
    outputs: (
        ProviderType[[InputType], OutputType]
        | ProviderType[[InputType, TraceType], OutputType]
    ) = Field(..., description="The outputs of the interaction.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="The metadata of the interaction."
    )

    _input_value_generator_provider: ValueGeneratorProvider[
        [TraceType], InputType, TraceType
    ] = PrivateAttr()
    _output_value_provider: ValueProvider[[InputType, TraceType], OutputType] = (
        PrivateAttr()
    )
    _resolved: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _validate_injection_mappings(
        self,
    ) -> "Interaction[InputType, OutputType, TraceType]":
        try:
            self._input_value_generator_provider = ValueGeneratorProvider.from_mapping(
                self.inputs, INJECTABLE_TRACE
            )
        except ValueError as e:
            raise ValueError(f"Error getting injection settings for inputs: {e}") from e

        try:
            self._output_value_provider = ValueProvider.from_mapping(
                self.outputs, INJECTABLE_INPUT, INJECTABLE_TRACE
            )
        except ValueError as e:
            raise ValueError(
                f"Error getting injection settings for outputs: {e}"
            ) from e

        if isinstance(
            self._input_value_generator_provider.provider, StaticValueProvider
        ) and isinstance(self._output_value_provider, StaticValueProvider):
            self._resolved = True

        return self

    @override
    async def generate(
        self, trace: TraceType
    ) -> AsyncGenerator[Interaction[InputType, OutputType, TraceType], TraceType]:
        if self._resolved:
            yield self  # type: ignore[misc]
            return

        generator = await self._input_value_generator_provider(trace)

        try:
            inputs = await anext(generator)
            while True:
                # Execute user-provided logic to transform inputs into either raw outputs
                # or a fully constructed Interaction instance.
                outputs = await self._output_value_provider(inputs, trace)
                # Yield the interaction back to the caller and wait for an updated trace
                # that captures the evaluation of this iteration.
                trace = yield self._get_interaction(
                    inputs,
                    cast(
                        OutputType | Interaction[InputType, OutputType, TraceType],
                        outputs,
                    ),
                )
                # Feed the updated trace to the input generator to produce the next inputs.
                inputs = await generator.asend(trace)
        except StopAsyncIteration:
            return
        finally:
            await generator.aclose()

    def _get_interaction(
        self,
        inputs: InputType,
        outputs: OutputType | Interaction[InputType, OutputType, TraceType],
    ) -> Interaction[InputType, OutputType, TraceType]:
        return (
            outputs
            if isinstance(outputs, Interaction)
            else Interaction(inputs=inputs, outputs=outputs, metadata=self.metadata)
        )

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield "Inputs: " + repr(self.inputs)
        yield "Outputs: " + repr(self.outputs)


__all__ = ["BaseInteraction", "Interaction"]
