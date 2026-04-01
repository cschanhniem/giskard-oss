from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

# from giskard.checks import (
#     CheckResult,
#     CheckStatus,
#     Metric,
#     ScenarioResult,
#     SuiteResult,
#     TestCaseResult,
#     Trace,
# )
from giskard.checks import (
    CheckResult,
    CheckStatus,
    Metric,
    ScenarioResult,
    SuiteResult,
    Trace,
)
from giskard.checks import (
    TestCaseResult as GiskardTestCaseResult,
)
from giskard.checks.export.junit import to_junit_xml


def _sample_suite_result() -> SuiteResult:
    return SuiteResult(
        results=[
            ScenarioResult(
                scenario_name="scenario_alpha",
                steps=[
                    GiskardTestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.PASS,
                                message="grounded",
                                metrics=[Metric(name="score", value=0.95)],
                                details={"check_name": "Groundedness"},
                            ),
                            CheckResult(
                                status=CheckStatus.FAIL,
                                message="answer is not grounded",
                                metrics=[Metric(name="confidence", value=0.2)],
                                details={"check_name": "Groundedness"},
                            ),
                        ],
                        duration_ms=125,
                    ),
                    GiskardTestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.ERROR,
                                message="judge crashed",
                                details={"check_name": "LLMJudge"},
                            )
                        ],
                        duration_ms=75,
                    ),
                    GiskardTestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.SKIP,
                                message="no retrieved context",
                                details={"check_name": "Conformity"},
                            )
                        ],
                        duration_ms=10,
                    ),
                ],
                duration_ms=210,
                final_trace=Trace(),
            )
        ],
        duration_ms=210,
    )


def test_to_junit_xml_builds_valid_xml() -> None:
    xml_string = to_junit_xml(_sample_suite_result())
    root = ET.fromstring(xml_string)

    assert root.tag == "testsuites"
    assert root.attrib["tests"] == "3"
    assert root.attrib["failures"] == "1"
    assert root.attrib["errors"] == "1"
    assert root.attrib["skipped"] == "1"
    assert root.attrib["time"] == "0.210"

    testsuite = root.find("testsuite")
    assert testsuite is not None
    assert testsuite.attrib["name"] == "scenario_alpha"
    assert testsuite.attrib["tests"] == "3"
    assert testsuite.attrib["failures"] == "1"
    assert testsuite.attrib["errors"] == "1"
    assert testsuite.attrib["skipped"] == "1"
    assert testsuite.attrib["time"] == "0.210"

    testcases = testsuite.findall("testcase")
    assert [testcase.attrib["name"] for testcase in testcases] == [
        "step_1",
        "step_2",
        "step_3",
    ]

    failure = testcases[0].find("failure")
    assert failure is not None
    assert failure.attrib["message"] == "answer is not grounded"
    assert "answer is not grounded" in (failure.text or "")

    error = testcases[1].find("error")
    assert error is not None
    assert error.attrib["message"] == "judge crashed"
    assert "judge crashed" in (error.text or "")

    skipped = testcases[2].find("skipped")
    assert skipped is not None
    assert "no retrieved context" in (skipped.text or "")


def test_to_junit_xml_includes_metric_properties_and_system_out() -> None:
    xml_string = to_junit_xml(_sample_suite_result())
    root = ET.fromstring(xml_string)

    first_testcase = root.find("./testsuite/testcase")
    assert first_testcase is not None

    properties = first_testcase.find("properties")
    assert properties is not None

    property_map = {
        prop.attrib["name"]: prop.attrib["value"]
        for prop in properties.findall("property")
    }
    assert property_map["check_1.score"] == "0.95"
    assert property_map["check_2.confidence"] == "0.2"

    system_out = first_testcase.find("system-out")
    assert system_out is not None
    assert "[PASS] Groundedness: grounded" in (system_out.text or "")
    assert "[FAIL] Groundedness: answer is not grounded" in (system_out.text or "")


def test_to_junit_xml_writes_file(tmp_path: Path) -> None:
    suite_result = _sample_suite_result()
    output_path = tmp_path / "test-results.xml"

    xml_string = to_junit_xml(suite_result, path=output_path)

    assert output_path.exists()
    assert ET.fromstring(xml_string).tag == "testsuites"
    assert ET.parse(output_path).getroot().tag == "testsuites"


def test_suite_result_convenience_method_matches_function() -> None:
    suite_result = _sample_suite_result()

    xml_from_method = suite_result.to_junit_xml()
    xml_from_function = to_junit_xml(suite_result)

    root_from_method = ET.fromstring(xml_from_method)
    root_from_function = ET.fromstring(xml_from_function)

    assert root_from_method.tag == root_from_function.tag
    assert root_from_method.attrib == root_from_function.attrib


def test_mixed_check_statuses_keep_testcase_status_unambiguous() -> None:
    suite_result = SuiteResult(
        results=[
            ScenarioResult(
                scenario_name="scenario_mixed",
                steps=[
                    GiskardTestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.FAIL,
                                message="hard failure",
                                details={"check_name": "CheckA"},
                            ),
                            CheckResult(
                                status=CheckStatus.SKIP,
                                message="skipped follow-up",
                                details={"check_name": "CheckB"},
                            ),
                        ],
                        duration_ms=50,
                    )
                ],
                duration_ms=50,
                final_trace=Trace(),
            )
        ],
        duration_ms=50,
    )

    root = ET.fromstring(to_junit_xml(suite_result))
    testcase = root.find("./testsuite/testcase")

    assert testcase is not None
    assert testcase.find("failure") is not None
    assert testcase.find("skipped") is None

    system_out = testcase.find("system-out")
    assert system_out is not None
    assert "[FAIL] CheckA: hard failure" in (system_out.text or "")
    assert "[SKIP] CheckB: skipped follow-up" in (system_out.text or "")
