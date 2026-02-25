"""Streaming events and async invocation with Portkey.

This example demonstrates how to consume streaming events from the agent
in real-time using ``stream_async``.  This is useful when you need to
display tokens as they arrive (e.g. in a chat UI) or when you want to
process model output incrementally.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/streaming.py
"""

import asyncio
import os

from strands import Agent

from strands_portkey import PortkeyModel

model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    model_id="gpt-4o-mini",
)

agent = Agent(
    model=model,
    system_prompt="You are a storyteller. Write vivid, short stories.",
)


async def main() -> None:
    """Run the agent with streaming output."""
    # stream_async yields events as they arrive from the model.
    print("--- Streaming tokens ---")
    async for event in agent.stream_async("Tell me a two-sentence story about a fox."):
        # TextStreamEvent contains a 'data' field with the text delta.
        if hasattr(event, "data"):
            print(event.data, end="", flush=True)
    print("\n")

    # invoke_async is the awaitable version of agent().
    print("--- Async invocation ---")
    result = await agent.invoke_async("Now continue that story with one more sentence.")
    print(result)


asyncio.run(main())
