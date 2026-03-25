"""Functional test configuration with auto-skip for missing SDKs."""

import importlib

import pytest

_PROVIDER_PACKAGES = {
    "openai": "openai",
    "google": "google.genai",
    "anthropic": "anthropic",
    "azure": "openai",
    "azure_ai": "openai",
}


def _is_installed(module_path: str) -> bool:
    try:
        importlib.import_module(module_path)
        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-skip tests marked with a provider whose SDK is not installed."""
    cache: dict[str, bool] = {}
    for item in items:
        for mark_name, package in _PROVIDER_PACKAGES.items():
            if mark_name in item.keywords:
                if package not in cache:
                    cache[package] = _is_installed(package)
                if not cache[package]:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Provider SDK '{package}' not installed"
                        )
                    )
