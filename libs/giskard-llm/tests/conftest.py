"""Test configuration with auto-skip for missing or present SDKs."""

import importlib

import pytest

_PROVIDER_PACKAGES = {
    "openai": "openai",
    "google": "google.genai",
    "anthropic": "anthropic",
    "azure": "openai",
    "azure_ai": "openai",
}

_ANY_PROVIDER_PACKAGES = ["openai", "google.genai", "anthropic"]


def _is_installed(module_path: str) -> bool:
    try:
        importlib.import_module(module_path)
        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-skip tests based on SDK availability.

    - Tests marked with a provider name skip when that SDK is missing.
    - Tests marked ``no_providers`` skip when any provider SDK is installed.
    """
    installed_cache: dict[str, bool] = {}
    for item in items:
        for mark_name, package in _PROVIDER_PACKAGES.items():
            if mark_name in item.keywords:
                if package not in installed_cache:
                    installed_cache[package] = _is_installed(package)
                if not installed_cache[package]:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Provider SDK '{package}' not installed"
                        )
                    )

    if "no_providers" not in {kw for item in items for kw in item.keywords}:
        return

    any_installed = any(_is_installed(p) for p in _ANY_PROVIDER_PACKAGES)
    if any_installed:
        for item in items:
            if "no_providers" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(
                        reason="no_providers tests require no provider SDKs installed"
                    )
                )
