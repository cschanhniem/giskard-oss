import json

import pytest
from rich.console import Console

_console = Console(force_terminal=True)


def pytest_generate_tests(metafunc):
    marker = metafunc.definition.get_closest_marker("eval")
    if marker:
        filename = marker.args[0]
        with open(filename, "r") as f:
            test_data = [json.loads(line) for line in f if line.strip()]
        metafunc.parametrize("data", test_data)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    This hook intercepts the test result. If it's a failure
    and the exception is 'Rich-aware', we render it.
    """
    outcome = yield
    report = outcome.get_result()

    # We only trigger on the 'call' phase (when the test actually runs)
    if report.failed and call.excinfo:
        # Check if the exception or the wrapped exception has __rich_console__
        exc = call.excinfo.value

        if hasattr(exc, "__rich_console__"):
            # Print a clear separator
            _console.print("\n")
            _console.print("[bold red]─[/bold red]" * 40)
            _console.print(exc)  # This triggers __rich_console__
            _console.print("[bold red]─[/bold red]" * 40)
