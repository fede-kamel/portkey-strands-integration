"""Custom callback handlers for real-time event processing.

Callback handlers receive events as the agent processes a request.
This is useful for custom logging, progress indicators, or piping
output to a UI.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/callback_handler.py
"""

import os

from strands import Agent, tool

from strands_portkey import PortkeyModel


@tool
def calculate(expression: str) -> dict:
    """Evaluate a math expression.

    Args:
        expression: Math expression to evaluate.

    Returns:
        Calculation result.
    """
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
    except Exception as exc:
        return {"status": "error", "content": [{"text": str(exc)}]}
    return {"status": "success", "content": [{"text": str(result)}]}


model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    model_id="gpt-4o-mini",
)


# ---------------------------------------------------------------------------
# 1. Custom callback that logs events
# ---------------------------------------------------------------------------
def logging_callback(**kwargs):
    """A callback handler that logs different event types."""
    if "data" in kwargs:
        # Text chunk from the model
        print(kwargs["data"], end="", flush=True)
    elif "current_tool_use" in kwargs:
        tool_use = kwargs["current_tool_use"]
        if tool_use.get("name"):
            print(f"\n[TOOL] Calling: {tool_use['name']}", flush=True)
    elif kwargs.get("init_event_loop"):
        print("[EVENT] Agent loop initialized", flush=True)
    elif kwargs.get("complete"):
        print("\n[EVENT] Response complete", flush=True)


print("=" * 60)
print("Custom callback handler")
print("=" * 60)

agent = Agent(
    model=model,
    tools=[calculate],
    callback_handler=logging_callback,
)

agent("What is 42 * 17? Use the calculator tool.")

# ---------------------------------------------------------------------------
# 2. Silent mode (no output) -- useful for programmatic access
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Silent mode (callback_handler=None)")
print("=" * 60)

silent_agent = Agent(
    model=model,
    callback_handler=None,
)

result = silent_agent("What is 2 + 2? Reply with just the number.")
print(f"Result (accessed programmatically): {result}")
