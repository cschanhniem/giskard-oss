import importlib
import os

import pytest
from giskard.agents.embeddings import EmbeddingModel
from giskard.agents.embeddings.base import EmbeddingParams
from giskard.agents.generators import Generator
from giskard.llm.routing import _provider_cache

_PROVIDER_PACKAGES = {
    "openai": "openai",
    "google": "google.genai",
    "anthropic": "anthropic",
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


@pytest.fixture(autouse=True)
def _clear_provider_cache():
    """Prevent stale async clients across event-loop boundaries."""
    _provider_cache.clear()
    yield
    _provider_cache.clear()


@pytest.fixture
async def generator():
    """Fixture providing a configured generator for tests."""
    return Generator(model=os.getenv("TEST_MODEL", "gemini/gemini-2.0-flash"))


@pytest.fixture
def embedding_model():
    """Fixture providing a configured embedding model for tests."""
    return EmbeddingModel(
        model=os.getenv("TEST_EMBEDDING_MODEL", "gemini/gemini-embedding-001"),
        params=EmbeddingParams(dimensions=1536),
    )
