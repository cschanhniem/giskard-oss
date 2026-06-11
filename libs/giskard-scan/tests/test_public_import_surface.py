import ast
from pathlib import Path

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
UPSTREAM_ROOTS = ("giskard.agents", "giskard.checks")


def _iter_python_files(root: Path):
    yield from sorted(root.rglob("*.py"))


def _violating_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(module.startswith(f"{root}.") for root in UPSTREAM_ROOTS):
                yield node.lineno, f"from {module} import ..."
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(f"{root}.") for root in UPSTREAM_ROOTS):
                    yield node.lineno, f"import {alias.name}"


def test_scan_src_imports_upstream_libs_only_from_package_roots():
    violations = [
        f"{path.relative_to(PACKAGE_SRC.parent)}:{line}: {statement}"
        for path in _iter_python_files(PACKAGE_SRC)
        for line, statement in _violating_imports(path)
    ]

    assert violations == []
