"""Functional tests for completion scenarios across all providers.

Design principle: every test must work with the weakest model available.
Assert on structure (non-empty, correct role, correct type), not content.

OpenTelemetry imports are lazy so ``pytest -m "not functional"`` does not require
OTEL packages, and CI can install only ``giskard-llm[$PROVIDER,$PROVIDER-otel]``.
"""

# pyright: reportMissingImports=false

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from giskard.checks.core.interaction.gen_ai import (
    GenAiTrace,
    Message,
    ModelResponse,
    ToolCallPart,
    ToolCallResponsePart,
)
from giskard.llm import ChatMessage, LLMClient, ToolDef

pytestmark = pytest.mark.functional

# openai-v2 instrumentation only treats the literal "true" as enabling capture for logger.emit()
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "SPAN_AND_EVENT"
os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"

# -- Provider parametrization -------------------------------------------------

_MODELS = {
    "openai": os.getenv("TEST_OPENAI_MODEL", "openai/gpt-4.1-nano"),
    "google": os.getenv("TEST_GOOGLE_MODEL", "google/gemini-2.0-flash"),
    "anthropic": os.getenv(
        "TEST_ANTHROPIC_MODEL", "anthropic/claude-haiku-4-5-20251001"
    ),
    "azure": os.getenv("TEST_AZURE_MODEL", "azure/gpt-4.1-nano"),
    "azure_ai": os.getenv("TEST_AZURE_AI_MODEL", "azure_ai/gpt-4.1-nano"),
}

_CONFIGURE_PARAMS = {
    "openai": {
        "provider": "openai",
        "api_key": "os.environ/OPENAI_API_KEY",  # pragma: allowlist secret
    },
    "google": {
        "provider": "google",
        "api_key": "os.environ/GOOGLE_API_KEY",  # pragma: allowlist secret
    },
    "anthropic": {
        "provider": "anthropic",
        "api_key": "os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
    },
    "azure": {
        "provider": "azure",
        "api_key": "os.environ/AZURE_API_KEY",  # pragma: allowlist secret
        "base_url": "os.environ/AZURE_API_BASE",  # pragma: allowlist secret
        "api_version": "os.environ/AZURE_API_VERSION",  # pragma: allowlist secret
    },
    "azure_ai": {
        "provider": "azure_ai",
        "api_key": "os.environ/AZURE_AI_API_KEY",  # pragma: allowlist secret
        "base_url": "os.environ/AZURE_AI_ENDPOINT",  # pragma: allowlist secret
    },
}

_PROVIDER_MARKS = {
    "openai": pytest.mark.openai,
    "google": pytest.mark.google,
    "anthropic": pytest.mark.anthropic,
    "azure": pytest.mark.azure,
    "azure_ai": pytest.mark.azure_ai,
}

_PROVIDER_PARAMS = [
    pytest.param(provider, marks=_PROVIDER_MARKS[provider], id=provider)
    for provider in _MODELS
]


@contextmanager
def _otel_in_memory_log_exporter(provider: str) -> Iterator[Any]:
    """Capture gen_ai log events (Anthropic instrumentation uses the Logs API when not legacy)."""
    try:
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import (
            InMemoryLogRecordExporter,
            SimpleLogRecordProcessor,
        )
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError as exc:
        pytest.skip(
            f"Install opentelemetry-sdk (e.g. giskard-llm[provider,provider-otel] extras). {exc}"
        )

    try:
        instrumentor = LLMClient.instrumentor_for(provider)
    except ImportError as exc:
        pytest.skip(str(exc))
    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))

    instrumentor.instrument(
        tracer_provider=TracerProvider(), logger_provider=logger_provider
    )
    try:
        yield log_exporter
    finally:
        instrumentor.uninstrument()
        logger_provider.shutdown()


def _gen_ai_events_from_log_exporter(log_exporter: Any) -> list[dict[str, Any]]:
    """Convert OTel log records to the shape expected by ``GenAiTrace.from_otel_logs``."""
    events: list[dict[str, Any]] = []
    for readable in log_exporter.get_finished_logs():
        lr = readable.log_record
        events.append({"event_name": lr.event_name, "body": lr.body})
    return events


def _make_client(provider: str) -> tuple[LLMClient, str]:
    """Create a configured LLMClient and return (client, model_string)."""
    client = LLMClient()
    alias = f"test-{provider}"
    client.configure(alias, **_CONFIGURE_PARAMS[provider])
    model = _MODELS[provider]
    _, model_name = model.split("/", 1)
    return client, f"{alias}/{model_name}"


# -- Message composition scenarios --------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_user_only(provider: str):
    """Single user message -> non-empty assistant response."""
    with _otel_in_memory_log_exporter(provider) as log_exporter:
        client, model = _make_client(provider)
        resp = await client.acompletion(
            model, [{"role": "user", "content": "Say hello"}]
        )

    assert len(resp.choices) > 0
    assert resp.choices[0].message.role == "assistant"

    events = _gen_ai_events_from_log_exporter(log_exporter)
    trace = GenAiTrace.from_otel(events, provider=provider)
    assert len(trace.interactions) == 1
    assert len(trace.interactions[0].inputs) == 1
    user_msg = trace.interactions[0].inputs[0]
    assert isinstance(user_msg, Message)
    assert user_msg.role == "user"
    assert user_msg.text == "Say hello"

    assert len(trace.interactions[0].outputs) == 1
    out0 = trace.interactions[0].outputs[0]
    assert isinstance(out0, ModelResponse)
    assert out0.role == "assistant"
    assert out0.text == resp.choices[0].message.content


WEATHER_TOOL: ToolDef = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    },
}


@pytest.mark.parametrize("provider", _PROVIDER_PARAMS)
async def test_completion_with_tools(provider: str):
    """Completion with tools -> non-empty assistant response."""
    with _otel_in_memory_log_exporter(provider) as log_exporter:
        client, model = _make_client(provider)
        resp = await client.acompletion(
            model,
            [{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[WEATHER_TOOL],
        )

        assert len(resp.choices) > 0
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].message.tool_calls is not None
        assert len(resp.choices[0].message.tool_calls) == 1
        assert resp.choices[0].message.tool_calls[0].type == "function"
        assert resp.choices[0].message.tool_calls[0].function.name == "get_weather"

        resp_two = await client.acompletion(
            model,
            [
                {"role": "user", "content": "What's the weather in London?"},
                ChatMessage(**resp.choices[0].message.model_dump()),
                {
                    "role": "tool",
                    "content": "The weather in London is sunny",
                    "tool_call_id": resp.choices[0].message.tool_calls[0].id,
                },
            ],
            tools=[WEATHER_TOOL],
        )

        assert len(resp_two.choices) > 0
        assert resp_two.choices[0].message.role == "assistant"
        assert isinstance(resp_two.choices[0].message.content, str)

    events = _gen_ai_events_from_log_exporter(log_exporter)
    trace = GenAiTrace.from_otel(events, provider=provider)

    assert len(trace.interactions) == 2
    assert len(trace.interactions[0].inputs) == 1
    first_user = trace.interactions[0].inputs[0]
    assert isinstance(first_user, Message)
    assert first_user.role == "user"
    assert first_user.text == "What's the weather in Paris?"

    # Second completion request: full message list (user + assistant tool call + tool result).
    assert len(trace.interactions[1].inputs) == 3
    second_user, assistant_call, tool_response = trace.interactions[1].inputs
    assert second_user.role == "user"
    assert second_user.text == "What's the weather in London?"

    assert assistant_call.role == "assistant"
    assert len(assistant_call.tool_calls) == 1
    tc0 = assistant_call.tool_calls[0]
    assert isinstance(tc0, ToolCallPart)
    assert tc0.name == "get_weather"
    assert tc0.id == resp.choices[0].message.tool_calls[0].id

    assert tool_response.role == "tool"
    assert len(tool_response.tool_responses) == 1
    tr0 = tool_response.tool_responses[0]
    assert isinstance(tr0, ToolCallResponsePart)
    assert tr0.id == resp.choices[0].message.tool_calls[0].id
    assert tr0.result == "The weather in London is sunny"

    assert len(trace.interactions[1].outputs) == 1
    out1 = trace.interactions[1].outputs[0]
    assert isinstance(out1, ModelResponse)
    assert out1.role == "assistant"
    assert out1.text == resp_two.choices[0].message.content
