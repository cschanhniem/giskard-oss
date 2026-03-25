import re
import sys
from unittest.mock import patch

from giskard.checks import CheckResult


def test_print_report_without_rich_falls_back_to_json(capsys):
    with patch.dict(sys.modules, {"rich": None}):
        check_result = CheckResult.success(message="Test message")
        check_result.print_report()

        captured = capsys.readouterr()

        assert check_result.model_dump_json(indent=2) in captured.out
        assert (
            'Rich is not installed; using JSON. Install the optional extra: pip install "giskard-checks[rich]"\n'
            in captured.out
        )


def test_print_report(capsys):
    check_result = CheckResult.failure(message="Test message")
    check_result.print_report()

    captured = capsys.readouterr()

    assert re.search(r"Unnamed check\s+FAIL\s+Test message", captured.out)
    assert (
        'Rich is not installed; using JSON. Install the optional extra: pip install "giskard-checks[rich]"\n'
        not in captured.out
    )
