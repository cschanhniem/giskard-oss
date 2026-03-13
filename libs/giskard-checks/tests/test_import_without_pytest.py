import subprocess
import sys


def test_giskard_checks_importable_without_pytest():
    code = r"""
import sys


class _BlockPytestFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pytest" or fullname.startswith("pytest."):
            raise ModuleNotFoundError(fullname)
        return None


sys.meta_path.insert(0, _BlockPytestFinder())
sys.modules.pop("pytest", None)

try:
    import pytest  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    raise SystemExit("pytest unexpectedly importable")

import giskard.checks  # noqa: F401
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
