import os
from contextlib import contextmanager
from pathlib import Path

import pytest
from giskard import agents
from giskard.checks import Scenario, Trace, set_default_generator
from giskard.checks import settings as checks_settings


# ------------ Fixtures ------------
# TODO: move this to giskard.checks.utils
# We want to reset the default generator after the tests are run to avoid any side effects.
@contextmanager
def with_generator(generator: agents.Generator):
    previous = checks_settings._default_generator
    set_default_generator(generator)
    try:
        yield generator
    finally:
        checks_settings._default_generator = previous


# TODO: make this a decorator @giskard.checks.pytest.parametrize(path)
def parametrize(path: Path, param_name: str = "scenario"):
    """Custom parametrize that loads Scenarios from a JSONL path.

    This is intentionally tolerant: missing files or invalid lines should not
    fail the whole test suite collection.
    """
    # TODO: make sure that we raise ImportError if pytest is not installed here.
    # we don't want pytest to be a dependency of giskard-checks.
    # We only support JSONL files for now. We could support other formats in the future.
    # We should also handle errors here (file not found, invalid JSON, etc.).
    with open(path, "r", encoding="utf-8") as f:
        scenarios: list[Scenario[object, object, Trace[object, object]]] = [
            Scenario.model_validate_json(line) for line in f if line.strip()
        ]

    # We return the standard pytest parametrize, but pre-filled
    return pytest.mark.parametrize(
        param_name, scenarios, ids=[s.name for s in scenarios]
    )


# ------------ End of Fixtures ------------


async def _sut(inputs):
    scenario = Scenario(**inputs)
    return await scenario.run()


@pytest.mark.functional
@parametrize(Path(__file__).parent / "dataset" / "conformity.jsonl")
async def test_conformity(scenario: Scenario[object, object, Trace[object, object]]):
    # Remove this condition to see what happens when a scenario fails.
    # Note this require to run the test with the -v flag.
    if scenario.annotations.get("expect_failure", False):
        return pytest.skip("Skipping scenario that is expected to fail.")

    with with_generator(
        agents.Generator(model=os.getenv("TEST_MODEL", "gemini/gemini-2.0-flash"))
    ):
        result = await scenario.run(target=_sut)
        result.assert_passed()
