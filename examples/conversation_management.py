"""Conversation management strategies with Portkey.

Conversation managers control how the agent's context window is managed.
This is critical for long conversations that might exceed the model's
token limit.

Three built-in strategies:
- NullConversationManager: No management (for short conversations)
- SlidingWindowConversationManager: Keep the N most recent messages (default)
- SummarizingConversationManager: Summarize older messages

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/conversation_management.py
"""

import os

from strands import Agent
from strands.agent.conversation_manager import (
    NullConversationManager,
    SlidingWindowConversationManager,
)

from strands_portkey import PortkeyModel

model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    model_id="gpt-4o-mini",
)

# ---------------------------------------------------------------------------
# 1. Sliding Window (default) -- keeps the last N messages
# ---------------------------------------------------------------------------
print("=" * 60)
print("Sliding Window (window_size=10)")
print("=" * 60)

agent = Agent(
    model=model,
    conversation_manager=SlidingWindowConversationManager(
        window_size=10,
        should_truncate_results=True,
    ),
    system_prompt="You are concise. Reply in one sentence.",
    callback_handler=None,
)

# Simulate a longer conversation
for i in range(5):
    result = agent(f"Turn {i + 1}: Tell me fact #{i + 1} about space.")
    print(f"  Turn {i + 1}: {result}")

print(f"  Messages in history: {len(agent.messages)}")

# ---------------------------------------------------------------------------
# 2. Null Manager -- no management, full history retained
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Null Manager (no trimming)")
print("=" * 60)

agent2 = Agent(
    model=model,
    conversation_manager=NullConversationManager(),
    system_prompt="You are concise. Reply in one sentence.",
    callback_handler=None,
)

for i in range(5):
    result = agent2(f"Turn {i + 1}: Tell me fact #{i + 1} about the ocean.")
    print(f"  Turn {i + 1}: {result}")

print(f"  Messages in history: {len(agent2.messages)}")
