from collections.abc import AsyncGenerator
from typing import Any, Self, override

from giskard.agents.errors.workflow_errors import ModelRefusalError, WorkflowError
from giskard.llm.types import UserMessage
from pydantic import BaseModel, Field, model_validator

from ..core import Trace
from ..core.exceptions import InputGenerationException
from ..core.extraction import JSONPathStr, resolve
from ..core.input_generator import InputGenerator
from ..core.mixin import WithGeneratorMixin


class LiteralGeneratorOutput[T](BaseModel):
    schema_issue: str | None = Field(
        default=None,
        description=(
            "Schema issue preventing message generation (e.g. no string-like field). "
            "Set this instead of message when the schema cannot produce a user message."
        ),
    )
    message: T | None = Field(
        default=None,
        description="The translated or verbatim message.",
    )

    @model_validator(mode="after")
    def _validate_message_and_schema_issue(self) -> "LiteralGeneratorOutput[T]":
        if self.message is not None and self.schema_issue is not None:
            raise ValueError("'message' and 'schema_issue' cannot both be set")
        if self.message is None and self.schema_issue is None:
            raise ValueError("one of 'message' or 'schema_issue' must be set")
        return self


@InputGenerator.register("literal_generator")
class LiteralGenerator[TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    InputGenerator[TraceType], WithGeneratorMixin
):
    """Yields a fixed value, optionally translating it to a target language via LLM.

    When no target language is configured (or the input and target languages
    already match), the value is yielded unchanged (noop path).  When languages
    differ the value is sent to the LLM for translation using the
    ``giskard.checks::generators/literal.j2`` template.

    Parameters
    ----------
    value:
        The string or UserMessage to yield / translate.
    target_language:
        Static target language (e.g. ``"French"``).  Mutually exclusive with
        ``target_language_key``.
    target_language_key:
        JSONPath expression (starting with ``trace.``) that resolves the target
        language at call time from the trace.  Mutually exclusive with
        ``target_language``.
    input_language:
        Optional source language hint passed to the translation template.  When
        provided and equal to the resolved target language the noop path is
        taken.
    max_retries:
        Maximum number of retry attempts when the model refuses or a provider
        policy block (HTTP 400) is encountered.
    """

    value: str | UserMessage = Field(
        ..., description="The value to yield or translate."
    )
    target_language: str | None = Field(
        default=None,
        description="Static target language for translation.",
    )
    target_language_key: JSONPathStr | None = Field(
        default=None,
        description="JSONPath expression to resolve target language from trace.",
    )
    input_language: str | None = Field(
        default=None,
        description="Source language hint for the translation template.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description=(
            "Maximum number of retry attempts per turn when the model refuses "
            "(schema_issue) or the request is blocked by a provider policy (WorkflowError)."
        ),
    )

    @model_validator(mode="after")
    def _validate_target_language_exclusive(self) -> Self:
        if self.target_language is not None and self.target_language_key is not None:
            raise ValueError(
                "Cannot set both 'target_language' and 'target_language_key' — choose one."
            )
        return self

    def _resolve_target_language(self, trace: TraceType) -> str | None:
        if self.target_language is not None:
            return self.target_language
        if self.target_language_key is not None:
            result = resolve(trace, self.target_language_key)
            return str(result) if result is not None else None
        return None

    @override
    async def __call__(
        self, trace: TraceType, input_type: type[Any] | None = None
    ) -> AsyncGenerator[Any, TraceType]:
        output_type = input_type or str

        target_language = self._resolve_target_language(trace)

        # Noop: no language configured, or languages match AND value is already the right type
        if target_language is None or (
            self.input_language is not None
            and self.input_language == target_language
            and isinstance(self.value, output_type)
        ):
            yield self.value
            return

        # LLM translation path
        workflow = self._generator.template(
            "giskard.checks::generators/literal.j2"
        ).with_output(LiteralGeneratorOutput[output_type])

        inputs: dict[str, Any] = {
            "value": self.value,
            "target_language": target_language,
            "input_language": self.input_language,
        }

        last_exc: Exception | None = None
        output: LiteralGeneratorOutput[Any] | None = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await workflow.with_inputs(**inputs).run()
                candidate = result.output
                if not candidate.schema_issue:
                    output = candidate
                    break
                last_exc = InputGenerationException(
                    f"schema issue at attempt {attempt + 1}: {candidate.schema_issue}"
                )
            except WorkflowError as exc:
                cause = exc.__cause__
                is_refusal = isinstance(cause, ModelRefusalError)
                is_policy_block = getattr(
                    cause, "status_code", None
                ) == 400 and "invalid_request_error" in str(cause)
                if not (is_refusal or is_policy_block):
                    raise
                last_exc = exc

        if output is None or output.message is None:
            raise InputGenerationException(
                f"translation failed after {self.max_retries + 1} attempt(s)"
            ) from last_exc

        yield output.message
