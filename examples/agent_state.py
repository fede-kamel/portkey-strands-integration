"""Agent state management with Portkey.

Agent state is a key-value store that persists across turns but is NOT
sent to the model during inference.  It is useful for tracking metadata,
counters, and user preferences that tools can read and write.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/agent_state.py
"""

import os

from strands import Agent, tool
from strands.tools.tools import ToolContext

from strands_portkey import PortkeyModel


@tool(context=True)
def track_query(query: str, tool_context: ToolContext) -> dict:
    """Track a user query and increment the query counter.

    Args:
        query: The query to track.
        tool_context: Injected tool context with access to agent state.

    Returns:
        Tracking confirmation.
    """
    count = tool_context.agent.state.get("query_count") or 0
    count += 1
    tool_context.agent.state.set("query_count", count)

    history = tool_context.agent.state.get("query_history") or []
    history.append(query)
    tool_context.agent.state.set("query_history", history)

    return {
        "status": "success",
        "content": [{"text": f"Query #{count} tracked: {query}"}],
    }


@tool(context=True)
def get_stats(tool_context: ToolContext) -> dict:
    """Get query statistics from agent state.

    Args:
        tool_context: Injected tool context with access to agent state.

    Returns:
        Statistics summary.
    """
    count = tool_context.agent.state.get("query_count") or 0
    history = tool_context.agent.state.get("query_history") or []
    return {
        "status": "success",
        "content": [{"text": f"Total queries: {count}. History: {history}"}],
    }


model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    model_id="gpt-4o-mini",
)

# Initialize agent with some state
agent = Agent(
    model=model,
    tools=[track_query, get_stats],
    state={"user_name": "Alice", "query_count": 0, "query_history": []},
    system_prompt=(
        "You are a helpful assistant. Use track_query to log each user question, "
        "and get_stats when asked about statistics."
    ),
    callback_handler=None,
)

print("=" * 60)
print("Agent state management")
print("=" * 60)

# State is accessible directly
print(f"Initial state: {agent.state.get()}")

# Make some queries that update state via tools
result = agent("Track this query: What is Python?")
print(f"\nAfter turn 1: {agent.state.get('query_count')} queries tracked")

result = agent("Track this query: How does async work?")
print(f"After turn 2: {agent.state.get('query_count')} queries tracked")

result = agent("Show me the query statistics.")
print(f"After turn 3: {result}")

print(f"\nFinal state: {agent.state.get()}")
