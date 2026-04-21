from typing import Annotated, Any, TypeVar

from pydantic import AfterValidator, BeforeValidator
from pydantic import BaseModel


def _not_empty_str(v: str) -> str:
    if not v.strip():
        raise ValueError("String must not be empty")

    return v


NotEmptyStr = Annotated[str, AfterValidator(_not_empty_str)]


def _not_empty_list[T](v: list[T]) -> list[T]:
    if not v:
        raise ValueError("List must not be empty")

    return v


T = TypeVar("T")

NotEmptyList = Annotated[list[T], AfterValidator(_not_empty_list)]

def _coerce_to_json_schema(v: Any) -> Any:
    if isinstance(v, type) and issubclass(v, BaseModel):
        schema = v.model_json_schema()
        schema["additionalProperties"] = False
        return {
            "type": "json_schema",
            "json_schema": {
                "name": v.__name__,
                "strict": True,
                "schema": schema,
            },
        }

    return v

JsonSchema = Annotated[dict[str, Any], BeforeValidator(_coerce_to_json_schema)]