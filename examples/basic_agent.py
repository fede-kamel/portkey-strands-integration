"""Basic agent with Portkey model -- simple text conversation.

This example demonstrates the simplest way to use the Strands Agents SDK with
the Portkey AI gateway.  It creates a single agent backed by an LLM of your
choice (routed through Portkey) and runs a short, multi-turn conversation.

Prerequisites
-------------
1. A Portkey account and API key (https://app.portkey.ai/).
2. A *virtual key* that wraps your provider API key -- create one in the
   Portkey dashboard under "Virtual Keys".  Alternatively, you can pass the
   raw provider API key via the ``Authorization`` header together with a
   ``provider`` slug (see the ``multi_provider.py`` example).

Set the following environment variables before running:

    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/basic_agent.py
"""

import os

from strands import Agent

from strands_portkey import PortkeyModel

# ---------------------------------------------------------------------------
# 1. Create the Portkey-backed model
# ---------------------------------------------------------------------------
# ``client_args`` are forwarded directly to the ``AsyncPortkey`` constructor.
# At minimum you need an ``api_key`` and some way to authenticate with the
# upstream provider -- here we use a *virtual key*.
model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    # ``model_id`` tells Portkey which model to invoke on the provider side.
    model_id="gpt-4o-mini",
)

# ---------------------------------------------------------------------------
# 2. Create a Strands agent with a system prompt
# ---------------------------------------------------------------------------
agent = Agent(
    model=model,
    system_prompt=(
        "You are a friendly and concise assistant. Keep your responses short -- one to two sentences at most."
    ),
)

# ---------------------------------------------------------------------------
# 3. Run a simple conversation
# ---------------------------------------------------------------------------
# The agent is callable -- just pass a string prompt and it returns an
# ``AgentResult`` whose string representation is the assistant's text reply.

print("--- Turn 1 ---")
result = agent("What is the capital of France?")
print(result)

print("\n--- Turn 2 ---")
# The agent maintains conversation history automatically, so follow-up
# questions work as expected.
result = agent("And what is its population?")
print(result)

print("\n--- Turn 3 ---")
result = agent("Give me one fun fact about that city.")
print(result)
