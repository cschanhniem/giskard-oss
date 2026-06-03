from asyncio import TaskGroup
from collections.abc import Sequence
from typing import Any

import numpy as np
from giskard.checks.scenarios.suite import Suite

from .generators.base import ScenarioGenerator
from .registry import _normalize_generator, suite_generator_registry


async def _generate_scenarios(
    description: str,
    languages: list[str],
    generators: list[ScenarioGenerator],
    max_scenarios: int | None = None,
    seed: int = 42,
) -> list[Any]:
    rng = np.random.default_rng(seed)

    tasks = []
    async with TaskGroup() as task_group:
        if max_scenarios is not None and len(generators) > 0:
            counts = rng.multinomial(
                max_scenarios, np.ones(len(generators)) / len(generators)
            )
            child_rngs = rng.spawn(len(generators))
            for generator, n, child_rng in zip(generators, counts, child_rngs):
                tasks.append(
                    task_group.create_task(
                        generator.generate_scenario(
                            description, languages, int(n), child_rng
                        )
                    )
                )
        else:
            for generator in generators:
                tasks.append(
                    task_group.create_task(
                        generator.generate_scenario(description, languages)
                    )
                )

    return [scenario for task in tasks for scenario in task.result()]


async def generate_suite(
    description: str,
    languages: list[str],
    generators: Sequence[ScenarioGenerator | type[ScenarioGenerator]] | None = None,
    max_scenarios: int | None = None,
    seed: int = 42,
) -> Suite[Any, Any]:
    """Generate a test suite by running all registered (or supplied) generators."""
    if max_scenarios is not None and max_scenarios < 0:
        raise ValueError(f"max_scenarios must be non-negative, got {max_scenarios}")

    resolved = (
        [_normalize_generator(g) for g in generators]
        if generators is not None
        else suite_generator_registry.generators()
    )
    scenarios = await _generate_scenarios(
        description, languages, resolved, max_scenarios, seed
    )
    return Suite(name="Scenarios", scenarios=scenarios)
