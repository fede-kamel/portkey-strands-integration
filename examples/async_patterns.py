"""Async invocation patterns with Portkey.

Strands Agents supports full async operation for use in async web servers,
concurrent workloads, or any application that already uses an event loop.

Patterns covered:

1. ``invoke_async`` -- awaitable agent call, returns AgentResult.
2. ``stream_async`` -- async generator, yields typed event objects.
3. Concurrent calls -- run multiple agents in parallel with asyncio.gather.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/async_patterns.py
"""

import asyncio
import os

from strands import Agent

from strands_portkey import PortkeyModel

PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY")
PORTKEY_VIRTUAL_KEY = os.environ.get("PORTKEY_VIRTUAL_KEY")


def make_agent(system_prompt="You are a concise assistant."):
    """Create a new agent instance backed by PortkeyModel.

    Args:
        system_prompt: System prompt for the agent.

    Returns:
        A configured Agent instance.
    """
    model = PortkeyModel(
        client_args={
            "api_key": PORTKEY_API_KEY,
            "virtual_key": PORTKEY_VIRTUAL_KEY,
        },
        model_id="gpt-4o-mini",
    )
    return Agent(model=model, system_prompt=system_prompt, callback_handler=None)


async def main() -> None:
    """Demonstrate async agent invocation patterns."""
    # -----------------------------------------------------------------------
    # 1. invoke_async -- simple awaitable call
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Pattern 1: invoke_async")
    print("=" * 60)

    agent = make_agent()
    result = await agent.invoke_async("What is the capital of Japan? One word.")
    print(f"Result: {result}")

    # -----------------------------------------------------------------------
    # 2. stream_async -- process tokens as they arrive
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pattern 2: stream_async (token-by-token)")
    print("=" * 60)

    agent = make_agent("You are a storyteller. Write vivid, short stories.")
    print("Story: ", end="", flush=True)
    async for event in agent.stream_async("Tell me a one-sentence story about a lighthouse."):
        if hasattr(event, "data"):
            print(event.data, end="", flush=True)
    print()

    # -----------------------------------------------------------------------
    # 3. Concurrent calls -- asyncio.gather for parallel agent invocations
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pattern 3: Concurrent agents with asyncio.gather")
    print("=" * 60)

    # Each agent is independent; they run concurrently.
    agents = [make_agent(f"You are expert #{i}. Be very brief.") for i in range(3)]
    questions = [
        "What is photosynthesis? One sentence.",
        "What is gravity? One sentence.",
        "What is entropy? One sentence.",
    ]

    results = await asyncio.gather(*[a.invoke_async(q) for a, q in zip(agents, questions, strict=True)])

    for q, r in zip(questions, results, strict=True):
        print(f"\nQ: {q}")
        print(f"A: {r}")


asyncio.run(main())
