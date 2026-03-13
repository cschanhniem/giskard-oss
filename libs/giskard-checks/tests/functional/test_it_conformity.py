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


async def _sut(inputs):
    return await Conformity(
        rule="The agent should not provide any financial advice."
    ).run(
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
    return (
        Scenario()
        .interact(data["conversation"], _sut)
        .check(
            Equals(
                expected_value="fail" if data["financial_advice"] else "pass",
                key="trace.last.outputs.status",
            )
        )
    )
