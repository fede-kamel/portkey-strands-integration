"""Lifecycle hooks for monitoring and modifying agent behavior.

Hooks let you subscribe to typed events throughout the agent lifecycle:
before/after invocations, model calls, and tool calls.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/hooks.py
"""

import os
import time

from strands import Agent, tool
from strands.agent.agent_hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)

from strands_portkey import PortkeyModel


@tool
def get_weather(city: str) -> dict:
    """Get weather for a city.

    Args:
        city: City name.

    Returns:
        Weather data.
    """
    return {"status": "success", "content": [{"text": f"Weather in {city}: 22°C, sunny"}]}


model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    model_id="gpt-4o-mini",
)

# ---------------------------------------------------------------------------
# 1. Individual callback registration
# ---------------------------------------------------------------------------
print("=" * 60)
print("Individual hook callbacks")
print("=" * 60)

agent = Agent(model=model, tools=[get_weather], callback_handler=None)


def on_before_model(event: BeforeModelCallEvent):
    """Log before each model call."""
    print("  [HOOK] Model call starting...")


def on_after_model(event: AfterModelCallEvent):
    """Log after each model call."""
    print(f"  [HOOK] Model call finished (stop_reason={event.stop_reason})")


def on_before_tool(event: BeforeToolCallEvent):
    """Log before each tool call."""
    print(f"  [HOOK] Tool call: {event.tool_use.get('name', 'unknown')}")


agent.hooks.add_callback(BeforeModelCallEvent, on_before_model)
agent.hooks.add_callback(AfterModelCallEvent, on_after_model)
agent.hooks.add_callback(BeforeToolCallEvent, on_before_tool)

result = agent("What is the weather in Paris?")
print(f"  Result: {result}\n")


# ---------------------------------------------------------------------------
# 2. HookProvider class -- group related hooks together
# ---------------------------------------------------------------------------
print("=" * 60)
print("HookProvider class")
print("=" * 60)


class TimingHook(HookProvider):
    """Measures time spent in model calls and tool calls."""

    def __init__(self):
        """Initialize timing hook."""
        self.model_start = 0.0
        self.total_model_time = 0.0
        self.tool_calls = 0

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register timing callbacks.

        Args:
            registry: The hook registry to register with.
        """
        registry.add_callback(BeforeInvocationEvent, self.on_start)
        registry.add_callback(BeforeModelCallEvent, self.on_model_start)
        registry.add_callback(AfterModelCallEvent, self.on_model_end)
        registry.add_callback(AfterToolCallEvent, self.on_tool_end)
        registry.add_callback(AfterInvocationEvent, self.on_end)

    def on_start(self, event: BeforeInvocationEvent):
        """Reset counters at invocation start."""
        self.total_model_time = 0.0
        self.tool_calls = 0

    def on_model_start(self, event: BeforeModelCallEvent):
        """Record model call start time."""
        self.model_start = time.time()

    def on_model_end(self, event: AfterModelCallEvent):
        """Accumulate model call duration."""
        self.total_model_time += time.time() - self.model_start

    def on_tool_end(self, event: AfterToolCallEvent):
        """Count tool calls."""
        self.tool_calls += 1

    def on_end(self, event: AfterInvocationEvent):
        """Print timing summary."""
        print(f"  [TIMING] Model time: {self.total_model_time:.2f}s")
        print(f"  [TIMING] Tool calls: {self.tool_calls}")


agent2 = Agent(
    model=model,
    tools=[get_weather],
    hooks=[TimingHook()],
    callback_handler=None,
)

result = agent2("What is the weather in Tokyo and London?")
print(f"  Result: {result}")
