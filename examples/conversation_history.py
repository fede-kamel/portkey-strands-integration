"""Multi-turn conversation with history management.

This example demonstrates how the Strands Agent manages conversation
history across turns, and how to pre-load messages for context.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/conversation_history.py
"""

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

# ---------------------------------------------------------------------------
# 1. Automatic history -- agent remembers previous turns
# ---------------------------------------------------------------------------
print("=" * 60)
print("Automatic conversation history")
print("=" * 60)

agent = Agent(
    model=model,
    system_prompt="You are a concise assistant. Reply in one sentence.",
)

result = agent("My favorite color is blue.")
print(f"Turn 1: {result}")

result = agent("What is my favorite color?")
print(f"Turn 2: {result}")

# ---------------------------------------------------------------------------
# 2. Pre-loaded history -- seed the conversation with prior messages
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Pre-loaded conversation history")
print("=" * 60)

# Start a new agent with existing conversation context.
agent_with_history = Agent(
    model=model,
    system_prompt="You are a concise assistant. Reply in one sentence.",
    messages=[
        {
            "role": "user",
            "content": [{"text": "I'm planning a trip to Japan in April."}],
        },
        {
            "role": "assistant",
            "content": [{"text": "April is a great time to visit Japan for cherry blossom season!"}],
        },
    ],
)

# The agent already knows about the trip context.
result = agent_with_history("What should I pack?")
print(f"With context: {result}")

# ---------------------------------------------------------------------------
# 3. Fresh agent -- no history carryover
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Fresh agent (no history)")
print("=" * 60)

fresh_agent = Agent(
    model=model,
    system_prompt="You are a concise assistant. Reply in one sentence.",
)

result = fresh_agent("What is my favorite color?")
print(f"No context: {result}")
