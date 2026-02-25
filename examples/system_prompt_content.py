"""Advanced system prompts with content blocks and cache points.

This example demonstrates using ``system_prompt_content`` instead of a
plain ``system_prompt`` string.  Content blocks allow you to include
multiple text segments and cache point hints for prompt caching.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/system_prompt_content.py
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
# 1. Simple system_prompt_content with multiple text blocks
# ---------------------------------------------------------------------------
# system_prompt_content is a list of SystemContentBlock objects.
# Each block can have a "text" key and an optional "cachePoint" key.
# Multiple text blocks are concatenated by the model provider.

print("=" * 60)
print("Multiple system prompt blocks")
print("=" * 60)

agent = Agent(
    model=model,
    system_prompt=[
        {"text": "You are a helpful coding assistant."},
        {"text": "Always include code examples in your answers."},
        {"text": "Keep explanations brief."},
    ],
)

result = agent("How do I reverse a list in Python?")
print(result)

# ---------------------------------------------------------------------------
# 2. System prompt with cache point hints
# ---------------------------------------------------------------------------
# Cache points tell the provider where to cache the system prompt prefix.
# This can reduce latency and cost for repeated calls with the same
# system prompt.  Not all providers support caching -- it's a hint.

print("\n" + "=" * 60)
print("System prompt with cache points")
print("=" * 60)

# A long system prompt that benefits from caching.
knowledge_base = (
    "Company policy: All returns must be processed within 30 days. "
    "Refunds are issued to the original payment method. "
    "Exchanges are available for items of equal or lesser value. "
    "Damaged items receive full refund plus shipping costs."
)

agent = Agent(
    model=model,
    system_prompt=[
        {"text": "You are a customer support agent."},
        {"text": knowledge_base},
        # Cache point after the static knowledge base.
        # Dynamic queries below won't bust the cache.
        {"cachePoint": {"type": "default"}},
        {"text": "Be empathetic and solution-oriented."},
    ],
)

# Multiple queries reuse the cached system prompt prefix.
for question in [
    "I want to return a shirt I bought last week.",
    "My order arrived damaged, what can I do?",
]:
    print(f"\nQ: {question}")
    result = agent(question)
    print(f"A: {result}")
