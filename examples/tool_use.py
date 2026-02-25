"""Agent with custom tools through Portkey.

This example shows how to define custom *tools* (functions the LLM can call)
and wire them into a Strands Agent that uses Portkey as its model provider.

The ``@tool`` decorator extracts metadata from the function signature,
type hints, and docstring to build an OpenAPI-compatible tool specification
that is sent to the model automatically.

Prerequisites
-------------
Set the following environment variables:

    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/tool_use.py
"""

import os
import random

from strands import Agent, tool

from strands_portkey import PortkeyModel

# ---------------------------------------------------------------------------
# 1. Define custom tools with the @tool decorator
# ---------------------------------------------------------------------------
# Each tool is a regular Python function.  The decorator inspects the
# docstring and type hints to generate the JSON schema the model needs.


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The name of the city to look up weather for.

    Returns:
        A dictionary with weather information.
    """
    # In a real application you would call an external weather API here.
    weather_data = {
        "city": city,
        "temperature_celsius": random.randint(-5, 35),
        "condition": random.choice(["sunny", "cloudy", "rainy", "snowy"]),
        "humidity_percent": random.randint(20, 90),
    }
    return {
        "status": "success",
        "content": [{"text": f"Weather in {city}: {weather_data}"}],
    }


@tool
def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate (e.g. "2 + 2").

    Returns:
        The result of the calculation.
    """
    # WARNING: eval() is used here only for demonstration purposes.
    # In production, use a safe math parser instead.
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
    except Exception as exc:
        return {
            "status": "error",
            "content": [{"text": f"Could not evaluate '{expression}': {exc}"}],
        }

    return {
        "status": "success",
        "content": [{"text": f"The result of {expression} is {result}"}],
    }


# ---------------------------------------------------------------------------
# 2. Set up the Portkey model
# ---------------------------------------------------------------------------
model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    model_id="gpt-4o-mini",
)

# ---------------------------------------------------------------------------
# 3. Create the agent and register the tools
# ---------------------------------------------------------------------------
agent = Agent(
    model=model,
    tools=[get_weather, calculate],
    system_prompt=(
        "You are a helpful assistant with access to a weather lookup tool and "
        "a calculator.  Use them whenever the user asks about weather or math."
    ),
)

# ---------------------------------------------------------------------------
# 4. Ask questions that require tool use
# ---------------------------------------------------------------------------

print("--- Weather question ---")
result = agent("What is the weather like in Tokyo right now?")
print(result)

print("\n--- Math question ---")
result = agent("What is 1337 * 42 + 7?")
print(result)

print("\n--- Combined question ---")
result = agent("If the temperature in Paris is above 20 degrees, calculate 20 * 3.14. Otherwise calculate 10 * 2.71.")
print(result)
