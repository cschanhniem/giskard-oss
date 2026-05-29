import numpy as np
import pytest
from giskard.checks.judges import LLMJudge
from giskard.checks.scenarios_generator.harmbench import HarmbenchScenarioGenerator


async def test_harmbench_generator_loads_scenarios():
    gen = HarmbenchScenarioGenerator()
    scenarios = await gen.generate_scenario(
        description="A general-purpose assistant", languages=["en"]
    )
    assert len(scenarios) > 0


async def test_harmbench_generator_injects_annotations():
    gen = HarmbenchScenarioGenerator()
    description = "A customer support chatbot"
    languages = ["en", "fr"]
    scenarios = await gen.generate_scenario(
        description=description, languages=languages
    )
    for scenario in scenarios:
        assert scenario.annotations.get("description") == description
        assert set(scenario.annotations.get("languages", [])) == set(languages)
        assert "semantic_category" in scenario.annotations
        assert "functional_category" in scenario.annotations


async def test_harmbench_generator_scenario_has_llm_judge():
    gen = HarmbenchScenarioGenerator()
    scenarios = await gen.generate_scenario(
        description="A general-purpose assistant", languages=["en"], max_scenarios=1
    )
    assert len(scenarios) == 1
    scenario = scenarios[0]
    checks = [check for step in scenario.steps for check in step.checks]
    assert len(checks) == 1
    assert isinstance(checks[0], LLMJudge)
    assert checks[0].prompt_path == "giskard.checks::scenarios/harmbench.j2"


async def test_harmbench_generator_max_scenarios_subsamples():
    gen = HarmbenchScenarioGenerator()
    rng = np.random.default_rng(42)
    scenarios = await gen.generate_scenario(
        description="A general-purpose assistant",
        languages=["en"],
        max_scenarios=5,
        rng=rng,
    )
    assert len(scenarios) == 5


async def test_harmbench_generator_max_scenarios_reproducible():
    gen = HarmbenchScenarioGenerator()
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    result_a = await gen.generate_scenario("desc", ["en"], max_scenarios=5, rng=rng_a)
    result_b = await gen.generate_scenario("desc", ["en"], max_scenarios=5, rng=rng_b)
    assert [s.name for s in result_a] == [s.name for s in result_b]


async def test_harmbench_generator_missing_file_raises(monkeypatch):
    from pathlib import Path

    import giskard.checks.scenarios_generator.harmbench as harmbench_mod

    monkeypatch.setattr(
        harmbench_mod, "_DATA_DIR", Path("/nonexistent/path/that/does/not/exist")
    )
    gen = HarmbenchScenarioGenerator()
    with pytest.raises(RuntimeError, match="not found"):
        await gen.generate_scenario("desc", ["en"])


def test_harmbench_generator_in_registry():
    from giskard.checks.scenarios_generator.harmbench import HarmbenchScenarioGenerator
    from giskard.checks.scenarios_generator.registry import suite_generator_registry

    types = {type(g) for g in suite_generator_registry.generators()}
    assert HarmbenchScenarioGenerator in types
