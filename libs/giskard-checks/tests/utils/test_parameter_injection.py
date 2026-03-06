"""Tests for parameter injection utilities."""

import typing
from typing import Any

import pytest
from giskard.checks.core.interaction.trace import Trace
from giskard.checks.utils.parameter_injection import (
    CallableInjectionMapping,
    ParameterInjectionRequirement,
)

INJECTABLE_TRACE = ParameterInjectionRequirement(class_info=Trace, optional=True)
INJECTABLE_INPUT = ParameterInjectionRequirement(class_info=Any, optional=True)


class TestParameterInjectionOptionalTrace:
    """Tests for injecting trace when typed as optional (Trace | None = None)."""

    def test_output_generator_receives_trace_when_typed_optional(self) -> None:
        """Trace is injected when parameter is typed as Trace | None = None.

        Regression test for https://github.com/Giskard-AI/giskard-oss/issues/2291
        """

        def output_generator(
            inputs: dict[str, Any], trace: Trace[Any, Any] | None = None
        ) -> dict[str, Any]:
            assert trace is not None
            return {"inputs": inputs, "count": len(trace.interactions)}

        mapping = CallableInjectionMapping.from_callable(
            output_generator, INJECTABLE_INPUT, INJECTABLE_TRACE
        )

        trace = Trace(interactions=[])
        injected = mapping.inject_parameters(output_generator, {"x": 1}, trace)
        result = injected()

        assert result == {"inputs": {"x": 1}, "count": 0}

    def test_inputs_generator_receives_trace_when_typed_optional(self) -> None:
        """Trace is injected for inputs when parameter is Trace | None = None."""

        def inputs_generator(trace: Trace[Any, Any] | None = None) -> str:
            assert trace is not None
            return f"count_{len(trace.interactions)}"

        mapping = CallableInjectionMapping.from_callable(
            inputs_generator, INJECTABLE_TRACE
        )

        trace = Trace(interactions=[])
        injected = mapping.inject_parameters(inputs_generator, trace)
        result = injected()

        assert result == "count_0"

    def test_int_or_trace_default_injected_with_trace(self) -> None:
        """Parameter typed as int | Trace = 1 receives trace when trace is provided."""

        def fn(x: int | Trace[Any, Any] = 1) -> int | Trace[Any, Any]:
            assert isinstance(x, Trace)
            return x

        mapping = CallableInjectionMapping.from_callable(fn, INJECTABLE_TRACE)

        trace = Trace(interactions=[])
        injected = mapping.inject_parameters(fn, trace)
        result = injected()

        assert result is trace

    def test_optional_param_without_matching_req_is_skipped(self) -> None:
        """Parameters with defaults that don't match any requirement use their default."""

        def fn(trace: Trace[Any, Any], other: int = 42) -> tuple[Trace[Any, Any], int]:
            return trace, other

        mapping = CallableInjectionMapping.from_callable(fn, INJECTABLE_TRACE)

        trace = Trace(interactions=[])
        injected = mapping.inject_parameters(fn, trace)
        result = injected()

        assert result[0] is trace
        assert result[1] == 42


class TestParameterInjectionDefaultParamsNotOverInjected:
    """Tests ensuring params with defaults are not incorrectly matched to already-satisfied requirements."""

    def test_typed_param_with_default_uses_default_when_req_already_satisfied(
        self,
    ) -> None:
        """Typed param (e.g. int) with default uses default; Any is not reused for it."""

        def fn(
            inputs: dict[str, Any], trace: Trace[Any, Any], multiplier: int = 2
        ) -> dict[str, Any]:
            return {
                "inputs": inputs,
                "count": len(trace.interactions) * multiplier,
            }

        mapping = CallableInjectionMapping.from_callable(
            fn, INJECTABLE_INPUT, INJECTABLE_TRACE
        )

        trace = Trace(interactions=[])
        injected = mapping.inject_parameters(fn, {"x": 1}, trace)
        result = injected()

        assert result["inputs"] == {"x": 1}
        assert result["count"] == 0 * 2  # multiplier=2 used from default

    def test_untyped_param_with_default_uses_default(self) -> None:
        """Untyped param with default uses default; not matched to any requirement."""

        def fn(trace: Trace[Any, Any], provided=42) -> tuple[Trace[Any, Any], int]:
            return trace, provided

        mapping = CallableInjectionMapping.from_callable(fn, INJECTABLE_TRACE)

        trace = Trace(interactions=[])
        injected = mapping.inject_parameters(fn, trace)
        result = injected()

        assert result[0] is trace
        assert result[1] == 42

    def test_extra_required_param_raises_when_all_reqs_satisfied(self) -> None:
        """Extra required param with no matching requirement raises TypeError."""

        def fn(
            inputs: dict[str, Any], trace: Trace[Any, Any], unmapped: str
        ) -> dict[str, Any]:
            return {"inputs": inputs, "unmapped": unmapped}

        with pytest.raises(TypeError, match="Parameter 'unmapped'.*no matching"):
            CallableInjectionMapping.from_callable(
                fn, INJECTABLE_INPUT, INJECTABLE_TRACE
            )


class TestParameterInjectionExistingBehavior:
    """Tests to ensure refactoring did not break existing behavior."""

    def test_required_param_without_match_raises(self) -> None:
        """Required parameters without matching requirement raise TypeError."""

        def fn(trace: Trace[Any, Any], unmapped: str) -> str:
            return unmapped

        with pytest.raises(TypeError, match="Parameter 'unmapped'.*no matching"):
            CallableInjectionMapping.from_callable(fn, INJECTABLE_TRACE)

    def test_trace_and_inputs_both_injected(self) -> None:
        """Both trace and inputs are injected when both params present."""

        def fn(inputs: dict[str, Any], trace: Trace[Any, Any]) -> dict[str, Any]:
            return {"inputs": inputs, "count": len(trace.interactions)}

        mapping = CallableInjectionMapping.from_callable(
            fn, INJECTABLE_INPUT, INJECTABLE_TRACE
        )

        trace = Trace(interactions=[])
        injected = mapping.inject_parameters(fn, {"x": 1}, trace)
        result = injected()

        assert result == {"inputs": {"x": 1}, "count": 0}

    def test_var_positional_skipped(self) -> None:
        """*args parameters are skipped."""

        def fn(*args: Any) -> str:
            return "ok"

        mapping = CallableInjectionMapping.from_callable(fn)
        injected = mapping.inject_parameters(fn)
        assert injected() == "ok"

    def test_var_keyword_skipped(self) -> None:
        """**kwargs parameters are skipped."""

        def fn(**kwargs: Any) -> str:
            return "ok"

        mapping = CallableInjectionMapping.from_callable(fn)
        injected = mapping.inject_parameters(fn)
        assert injected() == "ok"


class TestParameterInjectionRequirementMatching:
    """Tests for requirement matching and validation."""

    def test_trace_param_matches_trace_not_any_when_both_requirements_present(
        self,
    ) -> None:
        """Trace-typed param receives Trace, not Any, when both requirements exist."""

        # Order: INJECTABLE_INPUT first, INJECTABLE_TRACE second (as in Interact)
        def fn(
            inputs: dict[str, Any], trace: Trace[Any, Any] | None = None
        ) -> tuple[dict[str, Any], Trace[Any, Any]]:
            assert trace is not None
            return inputs, trace

        mapping = CallableInjectionMapping.from_callable(
            fn, INJECTABLE_INPUT, INJECTABLE_TRACE
        )

        inputs_val = {"x": 1}
        trace_val = Trace(interactions=[])
        injected = mapping.inject_parameters(fn, inputs_val, trace_val)
        result_inputs, result_trace = injected()

        assert result_inputs == inputs_val
        assert result_trace is trace_val

    def test_untyped_params_match_by_name_when_kwargs_reqs_used(self) -> None:
        """Untyped required params match by parameter name when kwargs_reqs provided."""

        def fn(trace, inputs) -> tuple[Any, Any]:
            return trace, inputs

        mapping = CallableInjectionMapping.from_callable(
            fn, trace=INJECTABLE_TRACE, inputs=INJECTABLE_INPUT
        )

        trace_val = Trace(interactions=[])
        inputs_val = {"key": "value"}
        # Args order matches param order: trace first, inputs second
        injected = mapping.inject_parameters(fn, trace_val, inputs_val)
        result_trace, result_inputs = injected()

        assert result_trace is trace_val
        assert result_inputs == inputs_val

    def test_typing_legacy_union_style_matches_requirement(self) -> None:
        """typing.Union (Union[X, Y]) is supported for requirement matching."""

        def fn(x: typing.Union[Trace[Any, Any], str]) -> Trace[Any, Any] | str:  # pyright: ignore[reportDeprecated]
            return x

        mapping = CallableInjectionMapping.from_callable(fn, INJECTABLE_TRACE)

        trace_val = Trace(interactions=[])
        injected = mapping.inject_parameters(fn, trace_val)
        assert injected() is trace_val

    def test_required_injection_unused_raises(self) -> None:
        """Required (non-optional) injection with no matching param raises TypeError."""
        required_trace = ParameterInjectionRequirement(class_info=Trace, optional=False)

        def fn(inputs: str) -> str:
            return inputs

        with pytest.raises(
            TypeError,
            match="Required injection for type.*Trace.*was not used by any parameter",
        ):
            CallableInjectionMapping.from_callable(fn, INJECTABLE_INPUT, required_trace)
