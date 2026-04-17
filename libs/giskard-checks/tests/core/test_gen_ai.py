"""Unit tests for ``GenAiTrace.from_otel_logs``."""

from giskard.checks.core.interaction.gen_ai import GenAiTrace, TextMessageLike


def test_from_otel_logs_multiple_choices_same_request():
    """Several ``gen_ai.choice`` events share the same input messages (e.g. ``n`` > 1)."""
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
    trace = GenAiTrace.from_otel_logs(events)
    assert len(trace.interactions) == 1
    assert len(trace.interactions[0].inputs) == 1
    assert trace.interactions[0].inputs[0] == TextMessageLike(
        role="user", content="Hello"
    )
    assert len(trace.interactions[0].outputs) == 2
    assert isinstance(trace.interactions[0].outputs[0].message, TextMessageLike)
    assert trace.interactions[0].outputs[0].message.content == "A"
    assert isinstance(trace.interactions[0].outputs[1].message, TextMessageLike)
    assert trace.interactions[0].outputs[1].message.content == "B"


def test_from_otel_logs_two_requests():
    """A new message after choices starts the next interaction."""
    events = [
        {
            "event_name": "gen_ai.user.message",
            "body": {"content": "First"},
        },
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
            "body": {"content": "Second"},
        },
        {
            "event_name": "gen_ai.choice",
            "body": {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Reply2"},
            },
        },
    ]
    trace = GenAiTrace.from_otel_logs(events)
    assert len(trace.interactions) == 2
    assert isinstance(trace.interactions[0].inputs[0], TextMessageLike)
    assert trace.interactions[0].inputs[0].content == "First"
    assert isinstance(trace.interactions[0].outputs[0].message, TextMessageLike)
    assert len(trace.interactions[0].outputs) == 1
    assert trace.interactions[0].outputs[0].message.content == "Reply1"
    assert isinstance(trace.interactions[1].inputs[0], TextMessageLike)
    assert trace.interactions[1].inputs[0].content == "Second"
    assert isinstance(trace.interactions[1].outputs[0].message, TextMessageLike)
    assert trace.interactions[1].outputs[0].message.content == "Reply2"
