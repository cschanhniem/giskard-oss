"""Functional tests for completion scenarios across all providers.

Design principle: every test must work with the weakest model available.
Assert on structure (non-empty, correct role, correct type), not content.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from giskard.checks.core.interaction.gen_ai import (
    ChoiceLike,
    GenAiTrace,
    TextMessageLike,
)
from giskard.llm import LLMClient
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider

pytestmark = pytest.mark.functional

# openai-v2 instrumentation only treats the literal "true" as enabling capture for logger.emit()
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

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

INSTRUMENTORS = {
    "openai": OpenAIInstrumentor(),
    "anthropic": AnthropicInstrumentor(use_legacy_attributes=False),
}


@contextmanager
def _otel_in_memory_log_exporter(instrumentor: BaseInstrumentor) -> Iterator[Any]:
    """Capture gen_ai log events (Anthropic instrumentation uses the Logs API when not legacy)."""

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
    if provider not in INSTRUMENTORS:
        pytest.skip(f"No instrumentor for provider: {provider}")

    with _otel_in_memory_log_exporter(INSTRUMENTORS[provider]) as log_exporter:
        client, model = _make_client(provider)
        resp = await client.acompletion(
            model, [{"role": "user", "content": "Say hello"}]
        )

    assert len(resp.choices) > 0
    assert resp.choices[0].message.role == "assistant"

    events = _gen_ai_events_from_log_exporter(log_exporter)
    trace = GenAiTrace.from_otel_logs(events)
    assert len(trace.interactions) == 1
    assert len(trace.interactions[0].inputs) == 1
    assert isinstance(trace.interactions[0].inputs[0], TextMessageLike)
    assert trace.interactions[0].inputs[0].role == "user"
    assert trace.interactions[0].inputs[0].content == "Say hello"

    assert len(trace.interactions[0].outputs) == 1
    out0 = trace.interactions[0].outputs[0]
    assert isinstance(out0, ChoiceLike)
    assert out0.message.role == "assistant"
    assert isinstance(out0.message, TextMessageLike)
    assert out0.message.content == resp.choices[0].message.content
