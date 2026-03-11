import pytest
from rich.console import Console

_console = Console(force_terminal=True)

# Mark every test in this directory as integration by default.
pytestmark = pytest.mark.functional


# TODO: make this a plugin
# # pyproject.toml
# [project.entry-points."pytest11"]
# giskard_checks = "giskard.checks.pytest.plugin"
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
