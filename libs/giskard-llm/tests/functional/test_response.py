"""Functional tests for ResponseProvider (Responses/Interactions API).

Only OpenAI and Google support these APIs. Each test is parametrized by provider
and asserts on structure (non-empty id, correct output types), not content.
"""

import os

import pytest
from giskard.llm import LLMClient
from giskard.llm.errors import UnsupportedOperationError
from giskard.llm.types import ResponseOutputFunctionCall

pytestmark = pytest.mark.functional

# -- Provider parametrization -------------------------------------------------

_RESPONSE_MODELS = {
    "openai": os.getenv("TEST_OPENAI_MODEL", "openai/gpt-4.1-nano"),
    "google": os.getenv("TEST_GOOGLE_MODEL", "google/gemini-2.0-flash"),
}

_CONFIGURE_PARAMS = {  # pragma: allowlist secret
    "openai": {"provider": "openai", "api_key": "os.environ/OPENAI_API_KEY"},
    "google": {"provider": "google", "api_key": "os.environ/GOOGLE_API_KEY"},
}

_PROVIDER_MARKS = {
    "openai": pytest.mark.openai,
    "google": pytest.mark.google,
}

_PROVIDER_PARAMS = [
    pytest.param(provider, marks=_PROVIDER_MARKS[provider], id=provider)
    for provider in _RESPONSE_MODELS
]


def _make_client(provider: str) -> tuple[LLMClient, str]:
    """Create a configured LLMClient and return (client, model_string)."""
    client = LLMClient()
    alias = f"test-{provider}"
    client.configure(alias, **_CONFIGURE_PARAMS[provider])
    model = _RESPONSE_MODELS[provider]
    _, model_name = model.split("/", 1)
    return client, f"{alias}/{model_name}"


# -- Basic response scenarios -------------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_text_input(provider: str):
    """String input -> non-empty text output with a valid id."""
    client, model = _make_client(provider)
    resp = await client.aresponse(model, "Say hello")
    assert resp.id
    assert len(resp.outputs) > 0
    assert resp.output_text is not None
    assert len(resp.output_text.strip()) > 0


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_with_instructions(provider: str):
    """Instructions param -> keyword appears in output."""
    client, model = _make_client(provider)
    resp = await client.aresponse(
        model,
        "Tell me something.",
        instructions="Always include the word PINEAPPLE in your response.",
    )
    assert resp.output_text is not None
    assert "pineapple" in resp.output_text.lower()


# -- Tool call scenario -------------------------------------------------------

_ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
}


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_function_call(provider: str):
    """Tools param with a function-triggering prompt -> function call output."""
    client, model = _make_client(provider)
    resp = await client.aresponse(
        model,
        "What is 2+2? Use the add tool.",
        tools=[_ADD_TOOL],
    )
    fc_outputs = [o for o in resp.outputs if isinstance(o, ResponseOutputFunctionCall)]
    assert len(fc_outputs) > 0
    fc = fc_outputs[0]
    assert fc.name == "add"
    assert "a" in fc.arguments and "b" in fc.arguments


# -- Stateful turn scenario ---------------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_respond_stateful_turn(provider: str):
    """Two calls with previous_id -> model remembers context."""
    client, model = _make_client(provider)

    resp1 = await client.aresponse(
        model,
        "My name is Zephyr. Remember that.",
        instructions="You are a helpful assistant with good memory.",
    )
    assert resp1.id

    resp2 = await client.aresponse(
        model,
        "What is my name?",
        previous_id=resp1.id,
    )
    assert resp2.output_text is not None
    assert "zephyr" in resp2.output_text.lower()


# -- Unsupported provider scenario --------------------------------------------


@pytest.mark.anthropic
async def test_respond_unsupported_provider():
    """Anthropic provider -> clear error when calling arespond()."""
    client = LLMClient()
    client.configure(
        "test-anthropic",
        provider="anthropic",
        api_key="os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
    )
    with pytest.raises(UnsupportedOperationError, match="does not support"):
        await client.aresponse("test-anthropic/claude-haiku-4-5-20251001", "Hello")
