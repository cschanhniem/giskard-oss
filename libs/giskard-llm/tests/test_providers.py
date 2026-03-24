"""Tests for provider response conversion logic using mocked SDK clients."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from giskard.llm.errors import RateLimitError
from giskard.llm.providers.openai import OpenAIProvider
from giskard.llm.routing import _provider_cache


@pytest.fixture(autouse=True)
def _clear_provider_cache():
    _provider_cache.clear()
    yield
    _provider_cache.clear()


# -- OpenAI provider ----------------------------------------------------------


def _make_openai_response(
    content: str | None = "Hello",
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
):
    """Build a mock that mimics openai ChatCompletion response."""
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


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_completion(mock_import):
    mock_openai = MagicMock()
    mock_import.return_value = mock_openai

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = MagicMock()
    provider._client.chat = MagicMock()
    provider._client.chat.completions = MagicMock()
    provider._client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response("Hello world")
    )

    resp = await provider.complete(
        "gpt-4o",
        [{"role": "user", "content": "Hi"}],
        temperature=0.5,
    )
    assert resp.choices[0].message.content == "Hello world"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.model == "gpt-4o"
    assert resp.usage is not None
    assert resp.usage.total_tokens == 15


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_completion_with_tool_calls(mock_import):
    mock_openai = MagicMock()
    mock_import.return_value = mock_openai

    tool_calls = [
        {
            "id": "call_1",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Paris"}),
            },
        }
    ]

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = MagicMock()
    provider._client.chat = MagicMock()
    provider._client.chat.completions = MagicMock()
    provider._client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response(
            content=None, finish_reason="tool_calls", tool_calls=tool_calls
        )
    )

    resp = await provider.complete("gpt-4o", [{"role": "user", "content": "Weather?"}])
    assert resp.choices[0].finish_reason == "tool_calls"
    assert resp.choices[0].message.tool_calls is not None
    assert resp.choices[0].message.tool_calls[0]["function"]["name"] == "get_weather"


@patch("giskard.llm.providers.openai._import_openai")
async def test_openai_rate_limit_error(mock_import):
    openai = pytest.importorskip("openai")
    mock_import.return_value = openai

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = MagicMock()
    provider._client.chat = MagicMock()
    provider._client.chat.completions = MagicMock()

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
    mock_openai = MagicMock()
    mock_import.return_value = mock_openai

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = MagicMock()
    provider._client.embeddings = MagicMock()
    provider._client.embeddings.create = AsyncMock(
        return_value=_make_openai_embedding_response([[0.1, 0.2], [0.3, 0.4]])
    )

    resp = await provider.embed("text-embedding-3-small", ["hello", "world"])
    assert len(resp.data) == 2
    assert resp.data[0].embedding == [0.1, 0.2]
    assert resp.data[1].embedding == [0.3, 0.4]


# -- Routing integration -------------------------------------------------------


@patch("giskard.llm.providers.openai._import_openai")
@patch("giskard.llm.providers.openai.OpenAIProvider.__init__", return_value=None)
async def test_acompletion_routes_to_openai(mock_init, mock_import):
    from giskard.llm import acompletion

    mock_openai = MagicMock()
    mock_import.return_value = mock_openai

    with patch.object(
        OpenAIProvider,
        "complete",
        new_callable=AsyncMock,
        return_value=MagicMock(choices=[]),
    ) as mock_complete:
        await acompletion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        mock_complete.assert_called_once_with(
            "gpt-4o", [{"role": "user", "content": "Hi"}]
        )
