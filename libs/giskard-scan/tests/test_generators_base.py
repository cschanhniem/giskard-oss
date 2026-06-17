from giskard.scan.generators.base import DatasetScenarioGenerator, ScenarioGenerator


class _Stub(ScenarioGenerator):
    async def generate_scenario(
        self,
        description,
        languages,
        max_scenarios=None,
        rng=None,
        target_mode="multiturn",
    ):
        return []


def test_scenario_generator_is_importable():
    gen = _Stub()
    assert isinstance(gen, ScenarioGenerator)


def test_scenario_generator_generate_scenario_accepts_target_mode():
    """generate_scenario must accept target_mode without raising TypeError."""
    import asyncio

    from giskard.scan.generators.base import ScenarioGenerator

    class _Stub(ScenarioGenerator):
        async def generate_scenario(
            self,
            description,
            languages,
            max_scenarios=None,
            rng=None,
            target_mode="multiturn",
        ):
            return []

    gen = _Stub()
    result = asyncio.get_event_loop().run_until_complete(
        gen.generate_scenario("desc", ["en"], target_mode="singleturn")
    )
    assert result == []


def test_dataset_generator_has_dataset_name_field():
    class _DS(DatasetScenarioGenerator):
        dataset_name: str = "stub"

    gen = _DS()
    assert gen.dataset_name == "stub"
