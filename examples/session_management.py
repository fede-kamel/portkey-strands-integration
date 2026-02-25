"""Session management for persistent conversations.

Session managers persist agent state and conversation history across
process restarts.  This example uses FileSessionManager to store
sessions on disk.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/session_management.py
"""

import os
import shutil
import tempfile

from strands import Agent
from strands.session.file_session_manager import FileSessionManager

from strands_portkey import PortkeyModel

model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    model_id="gpt-4o-mini",
)

# Use a temp directory for this demo; in production use a stable path.
storage_dir = tempfile.mkdtemp(prefix="strands_sessions_")
session_id = "demo-session-001"

print("=" * 60)
print(f"Session management (storage: {storage_dir})")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. First "process" -- create agent with session, have a conversation
# ---------------------------------------------------------------------------
print("\n--- First session ---")

session_manager = FileSessionManager(
    session_id=session_id,
    storage_dir=storage_dir,
)

agent = Agent(
    model=model,
    session_manager=session_manager,
    system_prompt="You are concise. Reply in one sentence.",
    callback_handler=None,
)

result = agent("My favorite programming language is Rust. Remember that.")
print(f"  Turn 1: {result}")

result = agent("What is 2 + 2?")
print(f"  Turn 2: {result}")

# Agent state and messages are now persisted to disk.
del agent

# ---------------------------------------------------------------------------
# 2. Second "process" -- restore from the same session
# ---------------------------------------------------------------------------
print("\n--- Restored session ---")

restored_session_manager = FileSessionManager(
    session_id=session_id,
    storage_dir=storage_dir,
)

restored_agent = Agent(
    model=model,
    session_manager=restored_session_manager,
    system_prompt="You are concise. Reply in one sentence.",
    callback_handler=None,
)

# The agent remembers the previous conversation.
result = restored_agent("What is my favorite programming language?")
print(f"  Turn 3 (after restore): {result}")

print(f"\n  Total messages in history: {len(restored_agent.messages)}")

# Cleanup
shutil.rmtree(storage_dir, ignore_errors=True)
print(f"  Cleaned up {storage_dir}")
