from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..core.result import CheckResult, ScenarioResult, SuiteResult, TestCaseResult


def _seconds(duration_ms: int) -> str:
    return f"{duration_ms / 1000:.3f}"


def _check_label(result: CheckResult, check_index: int) -> str:
    label = result.details.get("check_name") or result.details.get("check_kind")
    return str(label) if label else f"check_{check_index}"


def _check_summary(result: CheckResult, check_index: int) -> str:
    label = _check_label(result, check_index)
    return f"{label}: {result.message}" if result.message else label


def _ensure_properties(parent: ET.Element) -> ET.Element:
    properties = parent.find("properties")
    if properties is None:
        properties = ET.SubElement(parent, "properties")
    return properties


def _add_metric_properties(
    parent: ET.Element,
    result: CheckResult,
    check_index: int,
) -> None:
    if not result.metrics:
        return

    properties = _ensure_properties(parent)
    for metric in result.metrics:
        ET.SubElement(
            properties,
            "property",
            name=f"check_{check_index}.{metric.name}",
            value=str(metric.value),
        )


def _append_system_out(parent: ET.Element, test_case: TestCaseResult) -> None:
    if not test_case.results:
        return

    lines: list[str] = []
    for check_index, result in enumerate(test_case.results, start=1):
        lines.append(f"[{result.status.value.upper()}] {_check_summary(result, check_index)}")

    system_out = ET.SubElement(parent, "system-out")
    system_out.text = "\n".join(lines)


def _append_status_nodes(parent: ET.Element, test_case: TestCaseResult) -> None:
    if test_case.passed:
        return

    if test_case.failed:
        for check_index, result in enumerate(test_case.results, start=1):
            if not result.failed:
                continue
            node = ET.SubElement(
                parent,
                "failure",
                {
                    "type": _check_label(result, check_index),
                    "message": result.message or _check_label(result, check_index),
                },
            )
            node.text = _check_summary(result, check_index)
        return

    if test_case.errored:
        for check_index, result in enumerate(test_case.results, start=1):
            if not result.errored:
                continue
            node = ET.SubElement(
                parent,
                "error",
                {
                    "type": _check_label(result, check_index),
                    "message": result.message or _check_label(result, check_index),
                },
            )
            node.text = _check_summary(result, check_index)
        return

    skipped_messages = [
        _check_summary(result, check_index)
        for check_index, result in enumerate(test_case.results, start=1)
        if result.skipped
    ]
    node = ET.SubElement(parent, "skipped")
    node.text = "\n".join(skipped_messages)


def _scenario_counts(scenario: ScenarioResult[Any]) -> tuple[int, int, int, int]:
    tests = len(scenario.steps)
    failures = sum(1 for step in scenario.steps if step.failed)
    errors = sum(1 for step in scenario.steps if step.errored)
    skipped = sum(1 for step in scenario.steps if step.skipped)
    return tests, failures, errors, skipped


def _suite_counts(result: SuiteResult) -> tuple[int, int, int, int]:
    tests = sum(len(scenario.steps) for scenario in result.results)
    failures = sum(
        1 for scenario in result.results for step in scenario.steps if step.failed
    )
    errors = sum(
        1 for scenario in result.results for step in scenario.steps if step.errored
    )
    skipped = sum(
        1 for scenario in result.results for step in scenario.steps if step.skipped
    )
    return tests, failures, errors, skipped


def to_junit_xml(result: SuiteResult, path: str | Path | None = None) -> str:
    tests, failures, errors, skipped = _suite_counts(result)

    root = ET.Element(
        "testsuites",
        {
            "name": "giskard",
            "tests": str(tests),
            "failures": str(failures),
            "errors": str(errors),
            "skipped": str(skipped),
            "time": _seconds(result.duration_ms),
        },
    )

    for scenario in result.results:
        scenario_tests, scenario_failures, scenario_errors, scenario_skipped = (
            _scenario_counts(scenario)
        )

        testsuite_el = ET.SubElement(
            root,
            "testsuite",
            {
                "name": scenario.scenario_name,
                "tests": str(scenario_tests),
                "failures": str(scenario_failures),
                "errors": str(scenario_errors),
                "skipped": str(scenario_skipped),
                "time": _seconds(scenario.duration_ms),
            },
        )

        for step_index, test_case in enumerate(scenario.steps, start=1):
            testcase_el = ET.SubElement(
                testsuite_el,
                "testcase",
                {
                    "classname": scenario.scenario_name,
                    "name": f"step_{step_index}",
                    "time": _seconds(test_case.duration_ms),
                },
            )

            for check_index, check_result in enumerate(test_case.results, start=1):
                _add_metric_properties(testcase_el, check_result, check_index)

            _append_status_nodes(testcase_el, test_case)
            _append_system_out(testcase_el, test_case)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return ET.tostring(root, encoding="unicode")
