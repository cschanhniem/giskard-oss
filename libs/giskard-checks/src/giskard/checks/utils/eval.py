import inspect
import json
from functools import wraps

from ..core.scenario import Scenario

try:
    import pytest
except ImportError:
    pytest = None


def _require_pytest():
    if pytest is None:
        raise ImportError(
            "pytest is not installed. Please install it with `pip install pytest` to use eval_file()."
        )
    return pytest


def _eval_wrapper(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if inspect.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)

        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, Scenario):
            execution_result = await result.run()
            execution_result.assert_passed()
            return execution_result

        return result

    return wrapper


def eval_file(path):
    pytest = _require_pytest()

    def decorator(func):
        func = pytest.mark.eval(path)(func)
        wrapped = _eval_wrapper(func)
        return pytest.mark.asyncio(wrapped)

    return decorator


def pytest_generate_tests(metafunc):
    marker = metafunc.definition.get_closest_marker("eval")
    if marker:
        filename = marker.args[0]
        with open(filename, "r") as f:
            test_data = [json.loads(line) for line in f if line.strip()]
        metafunc.parametrize("data", test_data)
