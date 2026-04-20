from giskard.llm.types import (
    AssistantMessage,
    ChatCompletion,
    Choice,
    EmbeddingData,
    EmbeddingResponse,
    Function,
    FunctionCall,
    FunctionMessage,
    InputMessage,
    Message,
    RefusalContent,
    SystemMessage,
    TextContent,
    ToolMessage,
)
from pydantic import TypeAdapter

_MSG_ADAPTER = TypeAdapter(list[Message])


def test_chat_completion_model_dump():
    resp = ChatCompletion(
        choices=[
            Choice(
                message=AssistantMessage(content=[TextContent(text="Hello")]),
                finish_reason="stop",
            )
        ],
        model="gpt-4o",
    )
    dump = resp.model_dump()
    assert dump["choices"][0]["message"]["role"] == "assistant"
    assert dump["choices"][0]["message"]["content"] == [
        {"type": "text", "text": "Hello"}
    ]
    assert dump["choices"][0]["finish_reason"] == "stop"
    assert dump["model"] == "gpt-4o"


def test_assistant_message_excludes_none_by_default():
    msg = AssistantMessage(content=[TextContent(text="Hello")])
    dump = msg.model_dump()
    assert "tool_calls" not in dump


def test_assistant_message_text_property():
    msg = AssistantMessage(
        content=[TextContent(text="Hello"), TextContent(text="World")]
    )
    assert msg.text == "Hello\nWorld"
    assert msg.refusal is None


def test_assistant_message_refusal_coercion():
    msg = AssistantMessage.model_validate({"refusal": "No."})
    assert msg.refusal == "No."
    assert msg.text is None
    assert isinstance(msg.content[0], RefusalContent)  # pyright: ignore[reportOptionalSubscript]


def test_assistant_message_string_content_coerces_to_text_part():
    msg = AssistantMessage.model_validate({"content": "Hi"})
    assert msg.text == "Hi"
    assert isinstance(msg.content[0], TextContent)  # pyright: ignore[reportOptionalSubscript]


def test_assistant_message_includes_typed_tool_calls():
    msg = AssistantMessage(
        tool_calls=[
            FunctionCall(
                id="call_1",
                function=Function(name="add", arguments='{"a": 1, "b": 2}'),
            )
        ],
    )
    dump = msg.model_dump()
    assert dump["tool_calls"] is not None
    assert len(dump["tool_calls"]) == 1
    assert dump["tool_calls"][0]["function"]["name"] == "add"


def test_function_call_defaults_type_function():
    tc = FunctionCall(
        id="call_1",
        function=Function(name="get_weather", arguments='{"city": "Paris"}'),
    )
    assert tc.id == "call_1"
    assert tc.type == "function"
    assert tc.function.name == "get_weather"


def test_message_discriminator_user():
    msg = _MSG_ADAPTER.validate_python([{"role": "user", "content": "hi"}])[0]
    assert isinstance(msg, InputMessage)
    assert isinstance(msg.content, list)
    assert len(msg.content) == 1
    part = msg.content[0]
    assert isinstance(part, TextContent)
    assert part.text == "hi"


def test_message_discriminator_system():
    msg = _MSG_ADAPTER.validate_python([{"role": "system", "content": "be nice"}])[0]
    assert isinstance(msg, SystemMessage)


def test_message_discriminator_developer_routes_to_system():
    msg = _MSG_ADAPTER.validate_python([{"role": "developer", "content": "be nice"}])[0]
    assert isinstance(msg, SystemMessage)


def test_message_discriminator_tool():
    msg = _MSG_ADAPTER.validate_python(
        [{"role": "tool", "tool_call_id": "x", "content": "ok"}]
    )[0]
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "x"


def test_message_discriminator_function():
    msg = _MSG_ADAPTER.validate_python(
        [{"role": "function", "name": "f", "content": "ok"}]
    )[0]
    assert isinstance(msg, FunctionMessage)


def test_message_discriminator_assistant():
    msg = _MSG_ADAPTER.validate_python([{"role": "assistant", "content": "ack"}])[0]
    assert isinstance(msg, AssistantMessage)
    assert msg.text == "ack"


def test_embedding_response():
    resp = EmbeddingResponse(
        data=[
            EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0),
            EmbeddingData(embedding=[0.4, 0.5, 0.6], index=1),
        ]
    )
    assert len(resp.data) == 2
    assert resp.data[0].embedding == [0.1, 0.2, 0.3]
