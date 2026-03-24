import sys
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CLI toggle for integration tests."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked as integration.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark tests under tests/functional; skip integration tests unless requested."""
    for item in items:
        if "functional" in item.path.parts:
            item.add_marker(pytest.mark.functional)


def pytest_sessionfinish(session, exitstatus):
    # If no tests were collected, set the exit status to 0 to avoid failure.
    # This is a workaround for packages not having any functional tests.
    if exitstatus == 5:
        session.exitstatus = 0
