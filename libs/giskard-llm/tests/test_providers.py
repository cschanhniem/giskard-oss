"""Tests for provider response conversion, error mapping, and validation."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from giskard.llm.errors import BadRequestError, RateLimitError
from giskard.llm.providers.openai import OpenAIProvider
from giskard.llm.types import ToolCall

# -- OpenAI provider ----------------------------------------------------------


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


def _make_openai_provider():
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = MagicMock()
    provider._client.chat = MagicMock()
    provider._client.chat.completions = MagicMock()
    return provider


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
    assert json.loads(tcs[0].function.arguments) == {"city": "Paris"}


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
    provider._client.embeddings = MagicMock()
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
    from giskard.llm.providers.anthropic import AnthropicProvider

    mock_import.return_value = MagicMock()
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._merge_system = False
    provider._client = MagicMock()

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
    from giskard.llm.providers.anthropic import AnthropicProvider

    mock_anthropic = MagicMock()
    mock_import.return_value = mock_anthropic

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._merge_system = True
    provider._client = MagicMock()

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
    from giskard.llm.providers.anthropic import AnthropicProvider

    mock_import.return_value = MagicMock()
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._merge_system = False
    provider._client = MagicMock()

    with pytest.raises(BadRequestError, match="alternating"):
        await provider.complete(
            "claude-3",
            [
                {"role": "user", "content": "Hi"},
                {"role": "user", "content": "Hello again"},
            ],
        )
