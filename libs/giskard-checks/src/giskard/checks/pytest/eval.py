import inspect
from functools import wraps

from ..core.scenario import Scenario

try:
    import pytest
except ImportError:
    pytest = None


def _require_pytest():
    if not pytest:
        raise ImportError("pytest is required to use eval_file()")
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
