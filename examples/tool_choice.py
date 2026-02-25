"""Tool choice strategies with Portkey.

This example demonstrates how to control tool invocation behavior using
the ``tool_choice`` parameter:

- ``auto`` -- the model decides whether to call a tool (default).
- ``any`` -- the model *must* call at least one tool.
- ``tool`` -- the model *must* call a specific named tool.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/tool_choice.py
"""

import os
import random

from strands import Agent, tool

from strands_portkey import PortkeyModel


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The name of the city.

    Returns:
        Weather information.
    """
    return {
        "status": "success",
        "content": [{"text": f"Weather in {city}: {random.randint(15, 30)}°C, sunny"}],
    }


@tool
def get_time(timezone: str) -> dict:
    """Get the current time in a timezone.

    Args:
        timezone: The timezone name (e.g. "UTC", "US/Eastern").

    Returns:
        Current time information.
    """
    return {
        "status": "success",
        "content": [{"text": f"Current time in {timezone}: 14:30 UTC"}],
    }


model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    model_id="gpt-4o-mini",
)

# ---------------------------------------------------------------------------
# Strategy 1: auto (default) -- model decides whether to use tools
# ---------------------------------------------------------------------------
print("=" * 60)
print("Strategy: auto (model decides)")
print("=" * 60)

agent = Agent(model=model, tools=[get_weather, get_time])
result = agent("What's the weather in London?")
print(result)

# ---------------------------------------------------------------------------
# Strategy 2: any -- model MUST use at least one tool
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Strategy: any (must use a tool)")
print("=" * 60)

agent = Agent(
    model=model,
    tools=[get_weather, get_time],
    tool_choice={"any": {}},
)
result = agent("Hello, how are you?")
print(result)

# ---------------------------------------------------------------------------
# Strategy 3: tool -- model MUST use a specific tool
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Strategy: tool (must use get_weather)")
print("=" * 60)

agent = Agent(
    model=model,
    tools=[get_weather, get_time],
    tool_choice={"tool": {"name": "get_weather"}},
)
result = agent("What time is it in Tokyo?")
print(result)
