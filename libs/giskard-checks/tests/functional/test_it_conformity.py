from functools import partial
from pathlib import Path
from typing import Any

import pytest
from giskard.checks import (
    Conformity,
    Equals,
    Interaction,
    Scenario,
    Trace,
    eval_file,
)


async def _sut(inputs, rule: str):
    return await Conformity(rule=rule).run(
        Trace(
            interactions=[
                Interaction(
                    inputs=inputs[:-1],
                    outputs=inputs[-1],
                )
            ]
        )
    )


@pytest.mark.functional
@eval_file(
    Path(__file__).parent / "dataset" / "conformity.jsonl",
)
async def test_conformity(data: dict[str, Any]):
    if data.get("skip", False):
        return pytest.skip("Skipping due to skip flag.")

    return (
        Scenario()
        .interact(
            data["conversation"],
            partial(_sut, rule=data["rule"]),
        )
        .check(
            Equals(
                expected_value=data["expected_result"],
                key="trace.last.outputs.status",
            )
        )
    )
