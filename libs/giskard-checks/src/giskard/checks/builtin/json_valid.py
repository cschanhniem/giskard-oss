"""JSON validation check implementation."""

import json
from typing import Any, override

from jsonschema import SchemaError, ValidationError, validate
from pydantic import ConfigDict, Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, resolve
from ..core.result import CheckResult


@Check.register("json_valid")
class JsonValid[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Check that validates whether a trace value is valid JSON.

    The extracted value can be a JSON string or an already parsed JSON-compatible
    value such as a dict, list, string, number, boolean, or None.
    """

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the value to validate.",
    )
    schema_: dict[str, Any] | None = Field(
        default=None,
        alias="schema",
        description="Optional JSON Schema to validate the parsed JSON value against.",
    )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        value = resolve(trace, self.key)
        details: dict[str, Any] = {
            "key": self.key,
            "value": value,
            "schema": self.schema_,
        }

        if isinstance(value, NoMatch):
            return CheckResult.failure(
                message=f"No value found for key '{self.key}'.",
                details=details,
            )

        try:
            parsed_value = self._parse_json(value)
        except TypeError as err:
            return CheckResult.failure(
                message=str(err),
                details=details,
            )
        except json.JSONDecodeError as err:
            details["error"] = str(err)
            return CheckResult.failure(
                message=f"Value at key '{self.key}' is not valid JSON: {err}",
                details=details,
            )

        details["parsed_value"] = parsed_value

        if self.schema_ is not None:
            try:
                self._validate_schema(parsed_value)
            except SchemaError as err:
                details["error"] = err.message
                return CheckResult.error(
                    message=f"Provided JSON Schema is invalid: {err.message}.",
                    details=details,
                )
            except ValidationError as err:
                details["error"] = err.message
                return CheckResult.failure(
                    message=f"JSON value at key '{self.key}' does not match the provided schema: {err.message}.",
                    details=details,
                )

        return CheckResult.success(
            message=f"Value at key '{self.key}' is valid JSON.",
            details=details,
        )

    @staticmethod
    def _parse_json(value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)

        try:
            json.dumps(value)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Value is not JSON serializable: {err}") from err

        return value

    def _validate_schema(self, parsed_value: Any) -> None:
        validate(instance=parsed_value, schema=self.schema_)
