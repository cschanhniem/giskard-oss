"""Tests for the JsonValid check."""

from giskard.checks import Check, CheckStatus, Interaction, JsonValid, Trace
from giskard.checks.core.extraction import NoMatch


async def test_valid_json_string_passes() -> None:
    check = JsonValid()
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs='{"name": "Alice", "age": 30}')
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.passed
    assert result.details["parsed_value"] == {"name": "Alice", "age": 30}


async def test_invalid_json_string_fails_with_parse_error() -> None:
    check = JsonValid()
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs='{"name": "Alice"')
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.message is not None
    assert "not valid JSON" in result.message
    assert "line 1 column" in result.message
    assert "char" in result.message
    assert "error" in result.details


async def test_nested_jsonpath_extraction() -> None:
    check = JsonValid(key="trace.last.outputs.response")
    trace = await Trace.from_interactions(
        Interaction(
            inputs="Return JSON",
            outputs={"response": '{"items": [{"id": 1}, {"id": 2}]}'},
        )
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["parsed_value"] == {"items": [{"id": 1}, {"id": 2}]}


async def test_parsed_dict_passes() -> None:
    check = JsonValid()
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs={"name": "Alice", "age": 30})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["parsed_value"] == {"name": "Alice", "age": 30}


async def test_parsed_array_passes() -> None:
    check = JsonValid()
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs=[{"id": 1}, {"id": 2}])
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["parsed_value"] == [{"id": 1}, {"id": 2}]


async def test_schema_validation_passes() -> None:
    check = JsonValid(
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
    )
    trace = await Trace.from_interactions(
        Interaction(inputs="Return user data", outputs='{"name": "Alice", "age": 30}')
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.passed


async def test_schema_validation_fails() -> None:
    check = JsonValid(
        schema={
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["age"],
        }
    )
    trace = await Trace.from_interactions(
        Interaction(inputs="Return user data", outputs='{"age": "old"}')
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.message is not None
    assert "does not match the provided schema" in result.message
    assert result.details["error"] == "'old' is not of type 'integer'"


async def test_invalid_schema_returns_error() -> None:
    check = JsonValid(schema={"type": "not-a-json-schema-type"})
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs='{"name": "Alice"}')
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert result.message is not None
    assert "Provided JSON Schema is invalid" in result.message
    assert result.details["error"] == "'not-a-json-schema-type' is not valid under any of the given schemas"


async def test_missing_key_fails() -> None:
    check = JsonValid(key="trace.last.outputs.missing")
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs={"response": "{}"})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert isinstance(result.details["value"], NoMatch)
    assert result.message == "No value found for key 'trace.last.outputs.missing'."


async def test_non_serializable_value_fails() -> None:
    check = JsonValid()
    trace = await Trace.from_interactions(
        Interaction(inputs="Return JSON", outputs={"values": {1, 2, 3}})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.message == "Value is not JSON serializable: Object of type set is not JSON serializable"


def test_json_valid_is_exported() -> None:
    assert JsonValid.__name__ == "JsonValid"


def test_json_valid_serialization_roundtrip() -> None:
    check = JsonValid(key="trace.last.outputs.response", schema={"type": "object"})

    data = check.model_dump()
    restored = Check.model_validate(data)

    assert data["kind"] == "json_valid"
    assert data["schema"] == {"type": "object"}
    assert isinstance(restored, JsonValid)
    assert restored.key == "trace.last.outputs.response"
    assert restored.schema_ == {"type": "object"}
