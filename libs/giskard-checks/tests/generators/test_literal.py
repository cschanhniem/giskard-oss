import json
from collections.abc import Sequence
from typing import Any
from typing import override as type_override

import pytest
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import InputGenerationException
from giskard.checks.generators.literal import LiteralGenerator
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    Choice,
    CompletionResponse,
    UserMessage,
)

from .conftest import LLMTrace


def test_literal_generator_requires_value():
    with pytest.raises(Exception):
        LiteralGenerator()  # value is required  # pyright: ignore[reportCallIssue]


def test_literal_generator_rejects_both_target_language_fields():
    with pytest.raises(ValueError, match="target_language"):
        LiteralGenerator(
            value="hello",
            target_language="French",
            target_language_key="trace.annotations.lang",
        )


def test_literal_generator_accepts_no_language():
    gen = LiteralGenerator(value="hello")
    assert gen.target_language is None
    assert gen.target_language_key is None


def test_literal_generator_accepts_static_target_language():
    gen = LiteralGenerator(value="hello", target_language="French")
    assert gen.target_language == "French"


def test_literal_generator_accepts_target_language_key():
    gen = LiteralGenerator(value="hello", target_language_key="trace.annotations.lang")
    assert gen.target_language_key == "trace.annotations.lang"


def test_literal_generator_registered():
    from giskard.checks.core.input_generator import InputGenerator

    gen = InputGenerator.model_validate({"kind": "literal_generator", "value": "hi"})
    assert isinstance(gen, LiteralGenerator)


@pytest.mark.asyncio
async def test_noop_when_no_language_set_str_value():
    gen = LiteralGenerator(value="hello")
    trace = LLMTrace()
    results = [msg async for msg in gen(trace)]
    assert results == ["hello"]


@pytest.mark.asyncio
async def test_noop_when_no_language_set_user_message_value():
    msg_value = UserMessage(content="hello")
    gen = LiteralGenerator(value=msg_value)
    trace = LLMTrace()
    results = [msg async for msg in gen(trace, input_type=UserMessage)]
    assert results == [msg_value]


@pytest.mark.asyncio
async def test_noop_when_languages_match_and_type_matches():
    gen = LiteralGenerator(
        value="hello",
        target_language="English",
        input_language="English",
    )
    trace = LLMTrace()
    results = [msg async for msg in gen(trace)]
    assert results == ["hello"]


@pytest.mark.asyncio
async def test_noop_uses_target_language_key():
    gen = LiteralGenerator(
        value="hello",
        input_language="English",
        target_language_key="trace.annotations.lang",
    )
    # target resolved from annotations == input_language → noop
    trace = LLMTrace().model_copy(update={"annotations": {"lang": "English"}})
    results = [msg async for msg in gen(trace)]
    assert results == ["hello"]


@pytest.mark.asyncio
async def test_calls_llm_when_languages_differ():
    from .conftest import MockGenerator

    mock = MockGenerator(
        responses=[
            {"schema_issue": None, "message": "Bonjour"},
        ]
    )
    gen = LiteralGenerator(
        value="hello",
        target_language="French",
        input_language="English",
        generator=mock,
    )
    trace = LLMTrace()
    results = [msg async for msg in gen(trace)]
    assert results == ["Bonjour"]
    assert len(mock.calls) == 1


@pytest.mark.asyncio
async def test_calls_llm_when_input_language_not_set():
    from .conftest import MockGenerator

    mock = MockGenerator(
        responses=[
            {"schema_issue": None, "message": "Bonjour"},
        ]
    )
    gen = LiteralGenerator(
        value="hello",
        target_language="French",
        generator=mock,
    )
    trace = LLMTrace()
    results = [msg async for msg in gen(trace)]
    assert results == ["Bonjour"]
    assert len(mock.calls) == 1


def test_literal_generator_importable_from_generators():
    from giskard.checks.generators import LiteralGenerator as LG

    assert LG is not None


class _RefusingGenerator(BaseGenerator):
    refuse_times: int = 1
    valid_response: dict[str, Any]
    _calls: int = 0

    @type_override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self._calls += 1
        if self._calls <= self.refuse_times:
            return CompletionResponse(
                choices=[
                    Choice(
                        message=AssistantMessage(refusal="I can't help with that."),
                        finish_reason="refusal",
                        index=0,
                    )
                ]
            )
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content=json.dumps(self.valid_response)),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )


class _PolicyBlockGenerator(BaseGenerator):
    block_times: int = 1
    valid_response: dict[str, Any]
    _calls: int = 0

    @type_override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self._calls += 1
        if self._calls <= self.block_times:

            class _Err(Exception):
                status_code = 400

                def __str__(self):
                    return "invalid_request_error: content policy"

            raise _Err()
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content=json.dumps(self.valid_response)),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )


_VALID_TRANSLATED = {"schema_issue": None, "message": "Bonjour"}


@pytest.mark.asyncio
async def test_literal_generator_retries_refusal_and_succeeds():
    gen = _RefusingGenerator(refuse_times=1, valid_response=_VALID_TRANSLATED)
    lit = LiteralGenerator(
        value="hello",
        target_language="French",
        input_language="English",
        generator=gen,
        max_retries=1,
    )
    results = [msg async for msg in lit(LLMTrace())]
    assert results == ["Bonjour"]
    assert gen._calls == 2


@pytest.mark.asyncio
async def test_literal_generator_raises_after_all_retries_exhausted():
    gen = _RefusingGenerator(refuse_times=99, valid_response=_VALID_TRANSLATED)
    lit = LiteralGenerator(
        value="hello",
        target_language="French",
        input_language="English",
        generator=gen,
        max_retries=1,
    )
    with pytest.raises(InputGenerationException, match="translation failed"):
        async for _ in lit(LLMTrace()):
            pass
    assert gen._calls == 2  # 1 initial + 1 retry


@pytest.mark.asyncio
async def test_literal_generator_retries_policy_block_and_succeeds():
    gen = _PolicyBlockGenerator(block_times=1, valid_response=_VALID_TRANSLATED)
    lit = LiteralGenerator(
        value="hello",
        target_language="French",
        input_language="English",
        generator=gen,
        max_retries=1,
    )
    results = [msg async for msg in lit(LLMTrace())]
    assert results == ["Bonjour"]
    assert gen._calls == 2
