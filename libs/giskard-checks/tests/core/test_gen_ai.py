"""Unit tests for the ``gen_ai`` OpenTelemetry parsing package."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from giskard.checks.core.interaction.gen_ai import (
    EventStreamAdapter,
    GenAiTrace,
    IdentityNormalizer,
    Message,
    ModelResponse,
    SpanAttributesAdapter,
    TextPart,
    ToolCallPart,
    ToolCallResponsePart,
    ToolDefinition,
    detect_family,
    detect_provider,
    get_normalizer,
)
from giskard.checks.core.interaction.gen_ai.providers import AnthropicNormalizer

FIXTURES = Path(__file__).parent / "fixtures" / "gen_ai"


def _load(relpath: str) -> Any:
    with (FIXTURES / relpath).open(encoding="utf-8") as f:
        return json.load(f)


# -- Family A (event stream) --------------------------------------------------


def test_event_stream_multiple_choices_share_inputs():
    """Several ``gen_ai.choice`` events share the same input messages (``n`` > 1)."""
    events = [
        {
            "event_name": "gen_ai.user.message",
            "body": {"content": "Hello"},
        },
        {
            "event_name": "gen_ai.choice",
            "body": {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A"},
            },
        },
        {
            "event_name": "gen_ai.choice",
            "body": {
                "index": 1,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "B"},
            },
        },
    ]
    trace = GenAiTrace.from_otel(events)
    assert len(trace.interactions) == 1
    assert trace.interactions[0].inputs == [
        Message(role="user", parts=[TextPart(content="Hello")])
    ]
    assert [resp.text for resp in trace.interactions[0].outputs] == ["A", "B"]


def test_event_stream_next_message_starts_new_interaction():
    """A new message after choices starts the next interaction."""
    events = [
        {"event_name": "gen_ai.user.message", "body": {"content": "First"}},
        {
            "event_name": "gen_ai.choice",
            "body": {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Reply1"},
            },
        },
        {"event_name": "gen_ai.user.message", "body": {"content": "Second"}},
        {
            "event_name": "gen_ai.choice",
            "body": {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Reply2"},
            },
        },
    ]
    trace = GenAiTrace.from_otel(events)
    assert len(trace.interactions) == 2
    assert trace.interactions[0].inputs[0].text == "First"
    assert trace.interactions[0].outputs[0].text == "Reply1"
    assert trace.interactions[1].inputs[0].text == "Second"
    assert trace.interactions[1].outputs[0].text == "Reply2"


def test_event_stream_drop_redundant_input_history():
    """``drop_redundant_input_history`` trims repeated prefixes from inputs."""
    events = [
        {"event_name": "gen_ai.user.message", "body": {"content": "First"}},
        {
            "event_name": "gen_ai.choice",
            "body": {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Reply1"},
            },
        },
        {"event_name": "gen_ai.user.message", "body": {"content": "First"}},
        {
            "event_name": "gen_ai.assistant.message",
            "body": {"content": "Reply1"},
        },
        {"event_name": "gen_ai.user.message", "body": {"content": "Second"}},
        {
            "event_name": "gen_ai.choice",
            "body": {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Reply2"},
            },
        },
    ]
    raw = GenAiTrace.from_otel(events)
    assert len(raw.interactions[1].inputs) == 3
    assert raw.interactions[1].inputs[-1].text == "Second"

    deduped = GenAiTrace.from_otel(events, drop_redundant_input_history=True)
    assert len(deduped.interactions[1].inputs) == 1
    assert deduped.interactions[1].inputs[0].text == "Second"


def test_event_stream_trailing_inputs_without_choice_are_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events = [
        {"event_name": "gen_ai.user.message", "body": {"content": "First"}},
        {
            "event_name": "gen_ai.choice",
            "body": {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Reply1"},
            },
        },
        {
            "event_name": "gen_ai.user.message",
            "body": {"content": "Second — no completion logged"},
        },
    ]
    with caplog.at_level(logging.WARNING):
        trace = GenAiTrace.from_otel(events)
    assert len(trace.interactions) == 1
    assert trace.interactions[0].inputs[0].text == "First"
    assert any("gen_ai.choice" in r.message for r in caplog.records)


def test_event_stream_only_messages_no_choice_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events = [
        {"event_name": "gen_ai.user.message", "body": {"content": "Hello"}},
    ]
    with caplog.at_level(logging.WARNING):
        trace = GenAiTrace.from_otel(events)
    assert trace.interactions == []
    assert any("gen_ai.choice" in r.message for r in caplog.records)


def test_event_stream_tool_call_roundtrip_openai_shape():
    """OpenAI-shaped fixture parses into canonical parts."""
    events = _load("openai/events/tool_calls.json")
    trace = GenAiTrace.from_otel(events, provider="openai")

    assert len(trace.interactions) == 2

    # Second interaction carries the full history with an assistant tool-call
    # and a tool response.
    second = trace.interactions[1]
    assert [m.role for m in second.inputs] == ["user", "assistant", "tool"]
    assert second.inputs[0].text == "What's the weather in Paris?"

    assistant_call = second.inputs[1].tool_calls[0]
    assert isinstance(assistant_call, ToolCallPart)
    assert assistant_call.name == "get_weather"
    assert assistant_call.id == "call_VSPygqKTWdrhaFErNvMV18Yl"

    tool_response = second.inputs[2].tool_responses[0]
    assert isinstance(tool_response, ToolCallResponsePart)
    assert tool_response.id == "call_VSPygqKTWdrhaFErNvMV18Yl"
    assert tool_response.result == "rainy, 57°F"

    assert (
        second.outputs[0].text
        == "The weather in Paris is rainy and overcast, with temperatures around 57°F"
    )


def test_event_stream_raise_on_unknown_event_toggle():
    events = [{"event_name": "gen_ai.mystery.event", "body": {"x": 1}}]
    with pytest.raises(ValueError, match="Unknown event name"):
        GenAiTrace.from_otel(events)

    trace = GenAiTrace.from_otel(events, raise_on_unknown_event=False)
    assert trace.interactions == []


# -- Family A (Anthropic quirks) ----------------------------------------------


def test_anthropic_normalizer_handles_all_four_deviations():
    """End-to-end: Anthropic fixture covers every known deviation."""
    events = _load("anthropic/events/tool_calls.json")
    trace = GenAiTrace.from_otel(events, provider="anthropic")

    assert len(trace.interactions) == 2

    # (1) Tool definitions extracted from bogus user messages.
    assert trace.interactions[0].tool_definitions == [
        ToolDefinition(
            type="function",
            name="get_weather",
            description="Get the weather for a given city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    ]

    # (4) Top-level tool_calls in the first choice lifted into message.
    first_response = trace.interactions[0].outputs[0]
    assert first_response.role == "assistant"
    assert first_response.finish_reason == "tool_call"
    call = first_response.tool_calls[0]
    assert call.name == "get_weather"
    assert call.arguments == {"city": "Paris"}

    # (2) Assistant ``content`` with ``tool_use`` blocks maps to ToolCallPart.
    second = trace.interactions[1]
    assistant_history = second.inputs[1]
    assert assistant_history.role == "assistant"
    assert assistant_history.tool_calls[0].id == "toolu_01G6hjxH3Cmvya6DbX9aNxep"

    # (3) User ``content`` with ``tool_result`` blocks is reclassified to tool.
    tool_response_msg = second.inputs[2]
    assert tool_response_msg.role == "tool"
    assert tool_response_msg.tool_responses[0].id == "toolu_01G6hjxH3Cmvya6DbX9aNxep"
    assert (
        tool_response_msg.tool_responses[0].result == "The weather in London is sunny"
    )


def test_anthropic_normalizer_is_pure_on_events():
    normalizer = AnthropicNormalizer()
    events = _load("anthropic/events/tool_calls.json")
    original = json.loads(json.dumps(events))
    _ = normalizer.normalize_events(events)
    assert events == original


# -- Family B (span attributes) -----------------------------------------------


def test_span_attributes_parses_v1_40_shape():
    attrs = _load("openai/attributes/tool_calls.json")
    trace = GenAiTrace.from_otel(attrs, provider="openai")

    assert len(trace.interactions) == 1
    interaction = trace.interactions[0]
    assert interaction.system_instructions == [
        TextPart(content="You are a weather assistant.")
    ]

    assert [m.role for m in interaction.inputs] == ["user", "assistant", "tool"]
    call = interaction.inputs[1].tool_calls[0]
    assert call.name == "get_weather"
    assert call.arguments == {"location": "Paris"}
    assert interaction.inputs[2].tool_responses[0].result == "rainy, 57°F"

    assert interaction.outputs[0].text.startswith("The weather in Paris is currently")
    assert interaction.outputs[0].finish_reason == "stop"
    assert interaction.tool_definitions[0].name == "get_weather"


def test_span_attributes_accepts_json_string_attributes():
    """Span attributes may arrive JSON-encoded as strings."""
    attrs = _load("openai/attributes/tool_calls.json")
    as_strings = {
        key: json.dumps(value) if isinstance(value, (list, dict)) else value
        for key, value in attrs.items()
    }
    trace = GenAiTrace.from_otel(as_strings, provider="openai")
    assert len(trace.interactions) == 1
    assert trace.interactions[0].outputs[0].finish_reason == "stop"


# -- Detection ----------------------------------------------------------------


def test_detect_family_event_stream():
    events = [{"event_name": "gen_ai.user.message", "body": {"content": "hi"}}]
    assert detect_family(events) == "events"


def test_detect_family_attributes_from_dict():
    assert (
        detect_family({"gen_ai.input.messages": [], "gen_ai.output.messages": []})
        == "attributes"
    )


def test_detect_family_attributes_from_single_event():
    assert (
        detect_family(
            [
                {
                    "event_name": "gen_ai.client.inference.operation.details",
                    "attributes": {"gen_ai.input.messages": []},
                }
            ]
        )
        == "attributes"
    )


def test_detect_family_rejects_unknown_payload():
    with pytest.raises(ValueError, match="Unrecognized"):
        detect_family(42)


def test_detect_provider_from_attributes():
    assert detect_provider({"gen_ai.provider.name": "anthropic"}) == "anthropic"
    assert detect_provider({"gen_ai.system": "openai"}) == "openai"
    assert detect_provider({"foo": "bar"}) is None


# -- Wiring -------------------------------------------------------------------


def test_get_normalizer_returns_identity_for_unknown_provider():
    assert isinstance(get_normalizer(None), IdentityNormalizer)
    assert isinstance(get_normalizer("does-not-exist"), IdentityNormalizer)


def test_event_stream_adapter_can_be_used_directly():
    adapter = EventStreamAdapter()
    events = [
        {"event_name": "gen_ai.user.message", "body": {"content": "Hi"}},
        {
            "event_name": "gen_ai.choice",
            "body": {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello!"},
            },
        },
    ]
    [interaction] = adapter.parse(events, normalizer=IdentityNormalizer())
    assert interaction.inputs[0].text == "Hi"
    assert interaction.outputs[0].text == "Hello!"


def test_span_attributes_adapter_can_be_used_directly():
    attrs = _load("openai/attributes/tool_calls.json")
    adapter = SpanAttributesAdapter()
    [interaction] = adapter.parse(attrs, normalizer=IdentityNormalizer())
    assert isinstance(interaction.outputs[0], ModelResponse)
    assert interaction.outputs[0].finish_reason == "stop"
