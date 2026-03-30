"""Deprecated: import from giskard.agents.generators.giskard_llm_generator instead."""

import warnings

from .giskard_llm_generator import GiskardLLMGenerator, GiskardLLMRetryMiddleware

_DEPRECATED = {
    "LiteLLMGenerator": GiskardLLMGenerator,
    "LiteLLMRetryMiddleware": GiskardLLMRetryMiddleware,
}


def __getattr__(name: str):  # type: ignore[misc]
    if name in _DEPRECATED:
        warnings.warn(
            f"{name} is deprecated, use {_DEPRECATED[name].__name__} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
