"""Chatbot scan example using giskard.llm response API.

Demonstrates scanning a stateful chatbot (one string input per turn,
conversation state threaded via response id) with a generated suite.

Usage:
    OPENAI_API_KEY=sk-... uv run python examples/chatbot_scan/chatbot_scan.py
"""

import asyncio

from giskard.checks import Interaction, Trace, generate_suite
from giskard.llm import aresponse


async def chatbot(inputs: str, trace: Trace[str, str]) -> Interaction[str, str]:
    """Customer support chatbot backed by gpt-4o-mini via the response API."""
    previous_id = trace.last.metadata.get("response_id") if trace.last else None
    result = await aresponse(
        "openai/gpt-4o-mini",
        inputs,
        instructions="You are a helpful customer support agent for an e-commerce store.",
        previous_id=previous_id,
    )
    return Interaction(
        inputs=inputs,
        outputs=result.output_text or "",
        metadata={"response_id": result.id},
    )


async def main() -> None:
    print("Generating test suite...")
    suite = await generate_suite(
        description=(
            "A customer support chatbot for an e-commerce store "
            "that helps users with orders, returns, and product questions."
        ),
        languages=["en"],
        max_scenarios=5,
    )
    print(f"Generated {len(suite.scenarios)} scenarios. Running suite...\n")

    result = await suite.run(target=chatbot)
    result.print_report()
    print(f"\nPass rate: {result.pass_rate:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
