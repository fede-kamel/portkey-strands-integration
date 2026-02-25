"""Reasoning and extended thinking models via Portkey.

Some models perform explicit chain-of-thought reasoning before generating
their final response.  Portkey routes these requests like any other, and
PortkeyModel surfaces the reasoning tokens as ``reasoningContent`` stream
events through the Strands callback pipeline.

Two provider patterns are shown:

1. **OpenAI reasoning models** (o3-mini, o1) -- pass ``reasoning_effort``
   in ``params``.  The model exposes thinking tokens in
   ``delta.reasoning_content`` on the OpenAI-compat endpoint.

2. **Anthropic extended thinking** (claude-sonnet with thinking budget) --
   pass ``thinking`` in ``params``.  Portkey proxies the extended thinking
   parameter through to Anthropic.

Prerequisites
-------------
    # For OpenAI reasoning models:
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_OPENAI_VIRTUAL_KEY="your-openai-virtual-key"

    # For Anthropic extended thinking:
    export PORTKEY_ANTHROPIC_VIRTUAL_KEY="your-anthropic-virtual-key"

Usage:
    python examples/reasoning.py
"""

import os

from strands import Agent

from strands_portkey import PortkeyModel

PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY")

# ===================================================================
# 1. OpenAI reasoning models (o3-mini)
# ===================================================================
# OpenAI's o3-mini uses chain-of-thought reasoning internally.
# Set reasoning_effort to "low", "medium", or "high".

print("=" * 60)
print("OpenAI o3-mini with reasoning_effort")
print("=" * 60)

openai_virtual_key = os.environ.get("PORTKEY_OPENAI_VIRTUAL_KEY")
if openai_virtual_key:
    reasoning_model = PortkeyModel(
        client_args={
            "api_key": PORTKEY_API_KEY,
            "virtual_key": openai_virtual_key,
        },
        model_id="o3-mini",
        params={"reasoning_effort": "low"},
    )

    agent = Agent(
        model=reasoning_model,
        system_prompt="You are a problem solver. Show your reasoning.",
        callback_handler=None,
    )

    result = agent("If I have 3 apples and give away 1, then buy 5 more, how many do I have?")
    print(f"Answer: {result}")
else:
    print("Set PORTKEY_OPENAI_VIRTUAL_KEY to run this section.")

# ===================================================================
# 2. Anthropic extended thinking (claude-sonnet)
# ===================================================================
# Claude's extended thinking surfaces a <thinking> block before the
# final response.  Portkey proxies the thinking budget through.

print("\n" + "=" * 60)
print("Anthropic extended thinking via Portkey")
print("=" * 60)

anthropic_virtual_key = os.environ.get("PORTKEY_ANTHROPIC_VIRTUAL_KEY")
if anthropic_virtual_key:
    thinking_model = PortkeyModel(
        client_args={
            "api_key": PORTKEY_API_KEY,
            "virtual_key": anthropic_virtual_key,
        },
        model_id="claude-sonnet-4-20250514",
        params={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1000,
            },
        },
    )

    agent = Agent(
        model=thinking_model,
        callback_handler=None,
    )

    result = agent("What is the smallest prime number greater than 100?")
    print(f"Answer: {result}")
else:
    print("Set PORTKEY_ANTHROPIC_VIRTUAL_KEY to run this section.")
