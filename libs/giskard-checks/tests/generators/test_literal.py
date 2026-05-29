import pytest
from giskard.checks.generators.literal import LiteralGenerator
from giskard.llm.types import UserMessage

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
