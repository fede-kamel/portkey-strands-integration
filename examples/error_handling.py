"""Graceful error handling for context overflow and rate limiting.

PortkeyModel translates provider-specific error messages into Strands
exception types, letting you write provider-agnostic error handling code.

Two error types are demonstrated:

- ``ContextWindowOverflowException`` -- the prompt exceeds the model's
  context window.  Strategy: truncate or summarize the input and retry.
- ``ModelThrottledException`` -- the provider is rate-limiting your
  requests.  Strategy: back off with exponential delay and retry.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/error_handling.py
"""

import os
import time

from strands import Agent
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

from strands_portkey import PortkeyModel

model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    model_id="gpt-4o-mini",
)

# ---------------------------------------------------------------------------
# 1. Context window overflow -- truncate and retry
# ---------------------------------------------------------------------------
print("=" * 60)
print("Context window overflow handling")
print("=" * 60)

# Simulate a very long document that might overflow the context.
LONG_DOCUMENT = "Important fact: the sky is blue. " * 100

agent = Agent(model=model, callback_handler=None)

try:
    result = agent(f"Summarize this document: {LONG_DOCUMENT}")
    print(f"Success: {result}")
except ContextWindowOverflowException:
    print("Context overflow detected -- truncating document and retrying.")
    truncated = LONG_DOCUMENT[:500]
    result = agent(f"Summarize this document: {truncated}")
    print(f"Retry succeeded: {result}")

# ---------------------------------------------------------------------------
# 2. Rate limiting -- exponential back-off retry
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Rate limit (throttle) handling with retry")
print("=" * 60)


def call_with_retry(agent, prompt, max_retries=3, base_delay=1.0):
    """Call the agent with exponential back-off on throttling.

    Args:
        agent: The Strands Agent to call.
        prompt: The prompt string to send.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds between retries.

    Returns:
        The agent's response.

    Raises:
        ModelThrottledException: If all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return agent(prompt)
        except ModelThrottledException:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            print(f"  Throttled on attempt {attempt + 1}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    return None


result = call_with_retry(agent, "What is 2 + 2?")
print(f"Result: {result}")
