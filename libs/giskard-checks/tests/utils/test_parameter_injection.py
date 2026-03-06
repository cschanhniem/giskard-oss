import pytest
from giskard.checks.utils.parameter_injection import (
    CallableInjectionMapping,
    ParameterInjection,
    ParameterInjectionRequirement,
)


def test_inject_no_parameters():
    def func():
        return 1

    injection_mapping = CallableInjectionMapping.from_callable(func)

    assert injection_mapping.args == []
    assert injection_mapping.kwargs == {}

    assert injection_mapping.inject_parameters(func)() == 1


def test_inject_no_parameters_optional_requirement():
    def func():
        return 1

    injection_mapping = CallableInjectionMapping.from_callable(
        func, ParameterInjectionRequirement(class_info=int, optional=True)
    )

    assert injection_mapping.args == []
    assert injection_mapping.kwargs == {}

    assert injection_mapping.inject_parameters(func)() == 1


def test_inject_no_parameters_missing_required_requirement():
    def func():
        return 1

    with pytest.raises(
        TypeError, match=f"Parameter requirement {int} is required but not provided."
    ):
        _ = CallableInjectionMapping.from_callable(
            func, ParameterInjectionRequirement(class_info=int, optional=False)
        )


def test_inject_single_parameter():
    def func(a: int):
        return a

    injection_mapping = CallableInjectionMapping.from_callable(
        func, ParameterInjectionRequirement(class_info=int, optional=False)
    )

    assert injection_mapping.args == [
        ParameterInjection(position=0, name="a", class_info=int)
    ]
    assert injection_mapping.kwargs == {}

    assert injection_mapping.inject_parameters(func, 1)() == 1


def test_inject_single_parameter_with_default():
    def func(a: int = 0):
        return a

    injection_mapping = CallableInjectionMapping.from_callable(
        func, ParameterInjectionRequirement(class_info=int, optional=False)
    )

    assert injection_mapping.args == [
        ParameterInjection(position=0, name="a", class_info=int)
    ]
    assert injection_mapping.kwargs == {}

    assert injection_mapping.inject_parameters(func, 1)() == 1


def test_inject_not_provided_parameter():
    def func(a: int, b: str):
        return a, b

    with pytest.raises(
        TypeError,
        match=f"Parameter 'b' of type '{str}' is required but no matching `ParameterInjectionRequirement` was provided.",
    ):
        _ = CallableInjectionMapping.from_callable(
            func, ParameterInjectionRequirement(class_info=int, optional=False)
        )


def test_inject_ignore_missing_optional_parameter():
    def func(a: int, b: str = "default"):
        return a, b

    injection_mapping = CallableInjectionMapping.from_callable(
        func, ParameterInjectionRequirement(class_info=int, optional=True)
    )

    assert injection_mapping.args == [
        ParameterInjection(position=0, name="a", class_info=int)
    ]
    assert injection_mapping.kwargs == {}

    assert injection_mapping.inject_parameters(func, 1)() == (1, "default")


def test_inject_handle_type_union():
    def func(a: str | int | None = None):
        return a

    injection_mapping = CallableInjectionMapping.from_callable(
        func, ParameterInjectionRequirement(class_info=int, optional=True)
    )

    assert injection_mapping.args == [
        ParameterInjection(position=0, name="a", class_info=int)
    ]
    assert injection_mapping.kwargs == {}

    assert injection_mapping.inject_parameters(func, 1)() == 1
