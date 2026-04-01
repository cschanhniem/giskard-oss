"""Tests for provider response conversion, error mapping, and validation."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from giskard.llm.errors import BadRequestError, RateLimitError
from giskard.llm.providers.anthropic import AnthropicProvider
from giskard.llm.providers.base import (
    CompletionProvider,
    EmbeddingProvider,
    ResponseProvider,
)
from giskard.llm.providers.google import GoogleProvider
from giskard.llm.providers.openai import OpenAIProvider
from giskard.llm.types import (
    ResponseOutputFunctionCall,
    ResponseOutputText,
    ToolCall,
)

# -- Helpers -------------------------------------------------------------------


def _make_openai_provider():
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = MagicMock()
    provider._client.chat = MagicMock()
    provider._client.chat.completions = MagicMock()
    return provider


def _make_google_provider():
    provider = GoogleProvider.__new__(GoogleProvider)
    provider._client = MagicMock()
    return provider


def _make_anthropic_provider(merge_system: bool = False):
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._merge_system = merge_system
    provider._client = MagicMock()
    return provider


def _make_openai_response(
    content: str | None = "Hello",
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
):
    tc = None
    if tool_calls:
        tc = [
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in tool_calls
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tc,
                ),
                finish_reason=finish_reason,
                index=0,
            )
        ],
        model="gpt-4o",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


def _make_openai_embedding_response(embeddings: list[list[float]]):
    return SimpleNamespace(
        data=[
            SimpleNamespace(embedding=emb, index=i) for i, emb in enumerate(embeddings)
        ],
        model="text-embedding-3-small",
        usage=SimpleNamespace(prompt_tokens=8, total_tokens=8),
    )


def _make_openai_response_api_response(
    id: str = "resp_001",
    output_items: list[Any] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
):
    """Mock the OpenAI Responses API shape."""
    if output_items is None:
        output_items = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hello world")],
            )
        ]
    return SimpleNamespace(
        id=id,
        output=output_items,
        model="gpt-4o",
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def _make_google_interaction_response(
    id: str = "int_001",
    output_items: list[Any] | None = None,
    input_tokens: int = 8,
    output_tokens: int = 4,
):
    """Mock the Gemini Interactions API shape."""
    if output_items is None:
        output_items = [SimpleNamespace(type="text", text="Bonjour")]
    return SimpleNamespace(
        id=id,
        outputs=output_items,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


# -- OpenAI provider ----------------------------------------------------------


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_completion(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    provider._client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response("Hello world")
    )

    resp = await provider.complete(
        "gpt-4o", [{"role": "user", "content": "Hi"}], temperature=0.5
    )
    assert resp.choices[0].message.content == "Hello world"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.model == "gpt-4o"
    assert resp.usage is not None
    assert resp.usage.total_tokens == 15


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_completion_with_typed_tool_calls(mock_import):
    mock_import.return_value = MagicMock()
    tool_calls = [
        {
            "id": "call_1",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Paris"}),
            },
        }
    ]
    provider = _make_openai_provider()
    provider._client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response(
            content=None, finish_reason="tool_calls", tool_calls=tool_calls
        )
    )

    resp = await provider.complete("gpt-4o", [{"role": "user", "content": "Weather?"}])
    assert resp.choices[0].finish_reason == "tool_calls"
    tcs = resp.choices[0].message.tool_calls
    assert tcs is not None
    assert isinstance(tcs[0], ToolCall)
    assert tcs[0].function.name == "get_weather"
    assert tcs[0].function.arguments == '{"city": "Paris"}'


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_rate_limit_error(mock_import):
    openai = pytest.importorskip("openai")
    mock_import.return_value = openai

    provider = _make_openai_provider()
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": "rate limited"}}
    err = openai.RateLimitError(
        message="rate limited",
        response=mock_response,
        body={"error": {"message": "rate limited"}},
    )
    provider._client.chat.completions.create = AsyncMock(side_effect=err)

    with pytest.raises(RateLimitError) as exc_info:
        await provider.complete("gpt-4o", [{"role": "user", "content": "Hi"}])
    assert exc_info.value.status_code == 429


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_embedding(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    provider._client.embeddings = MagicMock()  # pyright: ignore[reportAttributeAccessIssue]
    provider._client.embeddings.create = AsyncMock(
        return_value=_make_openai_embedding_response([[0.1, 0.2], [0.3, 0.4]])
    )

    resp = await provider.embed("text-embedding-3-small", ["hello", "world"])
    assert len(resp.data) == 2
    assert resp.data[0].embedding == [0.1, 0.2]


# -- OpenAI message validation ------------------------------------------------


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_validate_empty_messages(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    with pytest.raises(BadRequestError, match="must not be empty"):
        await provider.complete("gpt-4o", [])


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_validate_system_only(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    with pytest.raises(BadRequestError, match="non-system message"):
        await provider.complete("gpt-4o", [{"role": "system", "content": "Be helpful"}])


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_validate_tool_missing_id(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    with pytest.raises(BadRequestError, match="tool_call_id"):
        await provider.complete(
            "gpt-4o",
            [
                {"role": "user", "content": "Hi"},
                {"role": "tool", "content": "result"},
            ],
        )


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_validate_empty_system_content(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    with pytest.raises(BadRequestError, match="non-empty content"):
        await provider.complete(
            "gpt-4o",
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Hi"},
            ],
        )


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_multiple_system_works(mock_import):
    """OpenAI supports multiple system messages natively."""
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    provider._client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response("Hello")
    )
    resp = await provider.complete(
        "gpt-4o",
        [
            {"role": "system", "content": "Be helpful"},
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert resp.choices[0].message.content == "Hello"


# -- Anthropic message validation ----------------------------------------------


@patch("giskard.llm.providers.anthropic._import_anthropic")
async def test_anthropic_validate_multi_system_raises(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_anthropic_provider()

    with pytest.raises(BadRequestError, match="multiple system messages"):
        await provider.complete(
            "claude-3",
            [
                {"role": "system", "content": "A"},
                {"role": "system", "content": "B"},
                {"role": "user", "content": "Hi"},
            ],
        )


@patch("giskard.llm.providers.anthropic._import_anthropic")
async def test_anthropic_validate_multi_system_with_merge(mock_import):
    mock_anthropic = MagicMock()
    mock_import.return_value = mock_anthropic

    provider = _make_anthropic_provider(merge_system=True)

    mock_raw = MagicMock()
    mock_raw.content = [SimpleNamespace(type="text", text="Hello")]
    mock_raw.stop_reason = "end_turn"
    mock_raw.model = "claude-3"
    mock_raw.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    provider._client.messages.create = AsyncMock(return_value=mock_raw)

    resp = await provider.complete(
        "claude-3",
        [
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert resp.choices[0].message.content == "Hello"


@patch("giskard.llm.providers.anthropic._import_anthropic")
async def test_anthropic_validate_alternation(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_anthropic_provider()

    with pytest.raises(BadRequestError, match="alternating"):
        await provider.complete(
            "claude-3",
            [
                {"role": "user", "content": "Hi"},
                {"role": "user", "content": "Hello again"},
            ],
        )


# -- OpenAI Responses API (respond) -------------------------------------------


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_respond_text(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    provider._client.responses = MagicMock()
    provider._client.responses.create = AsyncMock(
        return_value=_make_openai_response_api_response()
    )

    resp = await provider.respond("gpt-4o", "Hello")
    assert resp.id == "resp_001"
    assert len(resp.outputs) == 1
    assert isinstance(resp.outputs[0], ResponseOutputText)
    assert resp.outputs[0].text == "Hello world"
    assert resp.output_text == "Hello world"
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 10


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_respond_function_call(mock_import):
    mock_import.return_value = MagicMock()
    provider = _make_openai_provider()
    provider._client.responses = MagicMock()

    fc_item = SimpleNamespace(
        type="function_call",
        call_id="call_123",
        name="get_weather",
        arguments=json.dumps({"city": "Paris"}),
    )
    provider._client.responses.create = AsyncMock(
        return_value=_make_openai_response_api_response(output_items=[fc_item])
    )

    resp = await provider.respond("gpt-4o", "What's the weather?")
    assert len(resp.outputs) == 1
    assert isinstance(resp.outputs[0], ResponseOutputFunctionCall)
    assert resp.outputs[0].name == "get_weather"
    assert resp.outputs[0].arguments == {"city": "Paris"}
    assert resp.outputs[0].call_id == "call_123"


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_respond_error_mapping(mock_import):
    openai = pytest.importorskip("openai")
    mock_import.return_value = openai

    provider = _make_openai_provider()
    provider._client.responses = MagicMock()

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": "rate limited"}}
    err = openai.RateLimitError(
        message="rate limited",
        response=mock_response,
        body={"error": {"message": "rate limited"}},
    )
    provider._client.responses.create = AsyncMock(side_effect=err)

    with pytest.raises(RateLimitError) as exc_info:
        await provider.respond("gpt-4o", "Hello")
    assert exc_info.value.status_code == 429


# -- Google Interactions API (respond) -----------------------------------------


@patch("giskard.llm.providers.google._import_genai_errors")
async def test_google_respond_text(mock_errors):
    mock_errors.return_value = MagicMock()
    provider = _make_google_provider()
    provider._client.aio = MagicMock()
    provider._client.aio.interactions = MagicMock()
    provider._client.aio.interactions.create = AsyncMock(
        return_value=_make_google_interaction_response()
    )

    resp = await provider.respond("gemini-2.0-flash", "Hello")
    assert resp.id == "int_001"
    assert len(resp.outputs) == 1
    assert isinstance(resp.outputs[0], ResponseOutputText)
    assert resp.outputs[0].text == "Bonjour"


@patch("giskard.llm.providers.google._import_genai_errors")
async def test_google_respond_function_call(mock_errors):
    mock_errors.return_value = MagicMock()
    provider = _make_google_provider()
    provider._client.aio = MagicMock()
    provider._client.aio.interactions = MagicMock()

    fc_item = SimpleNamespace(
        type="function_call",
        name="get_weather",
        arguments={"city": "Tokyo"},
    )
    provider._client.aio.interactions.create = AsyncMock(
        return_value=_make_google_interaction_response(output_items=[fc_item])
    )

    resp = await provider.respond("gemini-2.0-flash", "Weather?")
    assert len(resp.outputs) == 1
    assert isinstance(resp.outputs[0], ResponseOutputFunctionCall)
    assert resp.outputs[0].name == "get_weather"
    assert resp.outputs[0].arguments == {"city": "Tokyo"}


# -- Protocol conformance checks -----------------------------------------------


def test_openai_implements_all_protocols():
    provider = _make_openai_provider()
    assert isinstance(provider, CompletionProvider)
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(provider, ResponseProvider)


def test_anthropic_implements_completion_only():
    provider = _make_anthropic_provider()
    assert isinstance(provider, CompletionProvider)
    assert not isinstance(provider, EmbeddingProvider)
    assert not isinstance(provider, ResponseProvider)


def test_google_implements_all_protocols():
    provider = _make_google_provider()
    assert isinstance(provider, CompletionProvider)
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(provider, ResponseProvider)
