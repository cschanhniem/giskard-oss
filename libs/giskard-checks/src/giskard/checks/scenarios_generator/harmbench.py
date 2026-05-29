import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.checks.generators import LiteralGenerator
from giskard.checks.judges import LLMJudge

from .base import ScenarioGenerator

_DATA_DIR = Path(__file__).parent / "data"


class HarmbenchScenarioGenerator(ScenarioGenerator):
    """Dataset-backed generator for HarmBench safety probes.

    Loads harmful prompts from the bundled ``harmbench.jsonl`` dataset and
    wraps each into a :class:`~giskard.checks.core.scenario.Scenario` with a
    single-turn interaction and a :class:`~giskard.checks.judges.LLMJudge`
    using the HarmBench evaluation criteria.

    Sampling behaviour mirrors
    :class:`~giskard.checks.scenarios_generator.base.DatasetScenarioGenerator`.
    """

    tags: ClassVar[list[str]] = [
        "gsk:threat-type='harmful-content-generation'",
    ]

    async def generate_scenario(
        self,
        description: str,
        languages: list[str],
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        """Build scenarios from the HarmBench probe dataset.

        Args:
            description: Natural-language description of the agent under test,
                forwarded to each scenario's annotations.
            languages: BCP-47 language codes forwarded to each scenario's
                annotations.
            max_scenarios: Maximum number of scenarios to return. When
                ``None``, the full dataset is returned.
            rng: Random generator used for subset sampling. A fresh
                ``np.random.default_rng()`` is created if ``None``.

        Returns:
            A list of annotated :class:`~giskard.checks.core.scenario.Scenario`
            objects ready to be collected into a
            :class:`~giskard.checks.scenarios.Suite`.
        """
        path = _DATA_DIR / "harmbench.jsonl"

        if not path.exists():
            raise RuntimeError(
                f"Dataset file not found: {path}. This may indicate a broken installation — try reinstalling the package."
            )

        _rng = rng if rng is not None else np.random.default_rng()

        scenarios = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                metadata = entry.get("metadata", {})
                target_language = (
                    languages[int(_rng.integers(len(languages)))] if languages else None
                )
                scenario = (
                    Scenario(
                        name=f"HarmBench — {metadata.get('semantic_category', 'unknown')}"
                    )
                    .interact(
                        inputs=LiteralGenerator(
                            value=entry["message"],
                            input_language="en",
                            target_language=target_language,
                        )
                    )
                    .check(
                        LLMJudge(prompt_path="giskard.checks::scenarios/harmbench.j2")
                    )
                    .with_annotations(
                        {
                            "description": description,
                            "languages": languages,
                            "semantic_category": metadata.get("semantic_category"),
                            "functional_category": metadata.get("functional_category"),
                            "tags": metadata.get("tags", []),
                        }
                    )
                )
                scenarios.append(scenario)

        if max_scenarios is not None and max_scenarios < len(scenarios):
            indices = _rng.choice(len(scenarios), size=max_scenarios, replace=False)
            scenarios = [scenarios[i] for i in sorted(indices)]

        return scenarios
