import functools
import inspect
from collections.abc import Callable
from types import UnionType
from typing import Any, cast, get_args, get_origin
from typing import TypeVar as TVar

from pydantic import BaseModel
from typing_extensions import TypeVar

R = TypeVar("R")


def _candidate_types_for_matching(annotation: Any) -> list[Any]:
    """Return types to check for requirement matching."""
    if isinstance(annotation, UnionType):
        return [a for a in get_args(annotation) if a is not type(None)]
    if annotation is inspect.Parameter.empty or annotation is Any:
        return []
    return [annotation]


def _type_matches_requirement(
    candidate_types: list[Any], req_class_info: type[Any] | Any
) -> bool:
    """True if any candidate type matches the requirement."""
    if req_class_info is Any:
        return True
    for candidate in candidate_types:
        if candidate is req_class_info:
            return True
        if (
            isinstance(candidate, type)
            and isinstance(req_class_info, type)
            and issubclass(cast(type, candidate), req_class_info)
        ):
            return True
    return False


def _find_matching_requirement(
    param_annotation: Any,
    candidate_types: list[Any],
    reqs_by_type: dict[type[Any] | Any, "ParameterInjectionRequirement"],
    satisfied_reqs: set[type[Any] | Any],
) -> "ParameterInjectionRequirement | None":
    """Find a requirement that matches the parameter's annotation."""
    for req_class_info, req in reqs_by_type.items():
        if req_class_info in satisfied_reqs:
            continue
        if param_annotation is inspect.Parameter.empty or param_annotation is Any:
            if req_class_info is Any:
                return req
        elif _type_matches_requirement(candidate_types, req_class_info):
            return req
    return None


def _find_requirement_for_untyped_required(
    reqs_by_type: dict[type[Any] | Any, "ParameterInjectionRequirement"],
    satisfied_reqs: set[type[Any] | Any],
) -> "ParameterInjectionRequirement | None":
    """Fallback: untyped required params can match any unsatisfied requirement."""
    for req_class_info, req in reqs_by_type.items():
        if req_class_info not in satisfied_reqs:
            return req
    return None


def _stored_class_info(
    param_annotation: Any, matching_req: "ParameterInjectionRequirement"
) -> Any:
    """Type to store in ParameterInjection; Pydantic rejects Union/GenericAlias."""
    if (
        isinstance(param_annotation, UnionType)
        or param_annotation is inspect.Parameter.empty
        or get_origin(param_annotation) is not None
    ):
        return matching_req.class_info
    return param_annotation


class ParameterInjectionRequirement(BaseModel, frozen=True):
    class_info: type[Any] | Any
    optional: bool = False


class ParameterInjection[T](BaseModel, frozen=True):
    position: int | None
    name: str | None
    class_info: type[T]

    def resolve(self, *args, **kwargs) -> T:
        if self.position is not None:
            return args[self.position]
        if self.name is not None:
            return kwargs[self.name]

        raise ValueError(f"ParameterInjection {self} has no position or name")


class CallableInjectionMapping(BaseModel, frozen=True):
    args: list[ParameterInjection[Any]]
    kwargs: dict[str, ParameterInjection[Any]]

    def inject_parameters(
        self, value: Callable[..., R], *args: Any, **kwargs: Any
    ) -> Callable[[], R]:
        return functools.partial(
            value,
            *[arg.resolve(*args, **kwargs) for arg in self.args],
            **{
                name: kwarg.resolve(*args, **kwargs)
                for name, kwarg in self.kwargs.items()
            },
        )

    @classmethod
    def from_callable(
        cls,
        callable: Callable[..., Any],
        *args_reqs: ParameterInjectionRequirement,
        **kwargs_reqs: ParameterInjectionRequirement,
    ) -> "CallableInjectionMapping":
        signature = inspect.signature(callable)
        reqs_by_type = {req.class_info: req for req in args_reqs}
        reqs_by_type.update({req.class_info: req for req in kwargs_reqs.values()})

        resolved_args: list[ParameterInjection[Any]] = []
        resolved_kwargs: dict[str, ParameterInjection[Any]] = {}
        satisfied_reqs: set[type[Any] | Any] = set()

        for idx, parameter in enumerate(signature.parameters.values()):
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            param_annotation = parameter.annotation
            if isinstance(param_annotation, TVar):
                param_annotation = param_annotation.__bound__ or Any

            candidate_types = _candidate_types_for_matching(param_annotation)

            matching_req = _find_matching_requirement(
                param_annotation, candidate_types, reqs_by_type, satisfied_reqs
            )
            if (
                matching_req is None
                and (
                    param_annotation is inspect.Parameter.empty
                    or param_annotation is Any
                )
                and parameter.default is inspect.Parameter.empty
            ):
                matching_req = _find_requirement_for_untyped_required(
                    reqs_by_type, satisfied_reqs
                )

            if matching_req is None:
                if parameter.default is not inspect.Parameter.empty:
                    continue
                raise TypeError(
                    f"Parameter '{parameter.name}' of type '{param_annotation}' is required "
                    "but no matching `ParameterInjectionRequirement` was provided."
                )

            stored_class_info = _stored_class_info(param_annotation, matching_req)
            is_positional = parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            injection = ParameterInjection(
                position=idx if is_positional else None,
                name=parameter.name,
                class_info=stored_class_info,
            )

            if injection.position is not None:
                resolved_args.append(injection)
            else:
                resolved_kwargs[parameter.name] = injection
            satisfied_reqs.add(matching_req.class_info)

        return cls(args=resolved_args, kwargs=resolved_kwargs)
