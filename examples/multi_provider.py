"""Using Portkey to switch between providers (OpenAI, Anthropic, etc.).

Portkey's AI gateway acts as a universal proxy.  By changing the
``client_args`` you can route the *same* Strands Agent to different providers
without touching any other code.  This example shows three common patterns:

1. **Virtual key** -- the recommended approach.  The provider API key is
   stored securely in Portkey's vault and referenced by a virtual key ID.
2. **Provider slug + direct API key** -- useful for quick experiments where
   you pass the provider name and API key directly.
3. **Config ID / inline config** -- Portkey configs let you define advanced
   routing strategies (fallbacks, load-balancing, retries) in the Portkey
   dashboard or inline.

Prerequisites
-------------
Set whichever environment variables you need for the patterns you want to try:

    # Pattern 1 -- virtual keys (create them in the Portkey dashboard)
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_OPENAI_VIRTUAL_KEY="vk-openai-..."
    export PORTKEY_ANTHROPIC_VIRTUAL_KEY="vk-anthropic-..."

    # Pattern 2 -- direct provider API keys
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."

    # Pattern 3 -- Portkey config ID
    export PORTKEY_CONFIG_ID="pc-..."

Usage:
    python examples/multi_provider.py
"""

import os

from strands import Agent

from strands_portkey import PortkeyModel

PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY")
PROMPT = "Explain quantum entanglement in one sentence."

# ===================================================================
# Pattern 1: Virtual keys (recommended for production)
# ===================================================================
# Virtual keys are stored in Portkey's vault.  You reference them by ID.
# This keeps raw provider API keys out of your code and environment.

print("=" * 60)
print("Pattern 1: Virtual Keys")
print("=" * 60)

openai_model = PortkeyModel(
    client_args={
        "api_key": PORTKEY_API_KEY,
        "virtual_key": os.environ.get("PORTKEY_OPENAI_VIRTUAL_KEY"),
    },
    model_id="gpt-4o-mini",
)

agent = Agent(
    model=openai_model,
    system_prompt="You are a physics tutor. Be concise.",
)
print("[OpenAI via virtual key]")
print(agent(PROMPT))

# Switch to Anthropic -- same agent structure, different model object.
anthropic_model = PortkeyModel(
    client_args={
        "api_key": PORTKEY_API_KEY,
        "virtual_key": os.environ.get("PORTKEY_ANTHROPIC_VIRTUAL_KEY"),
    },
    model_id="claude-sonnet-4-20250514",
)

agent = Agent(
    model=anthropic_model,
    system_prompt="You are a physics tutor. Be concise.",
)
print("\n[Anthropic via virtual key]")
print(agent(PROMPT))


# ===================================================================
# Pattern 2: Provider slug + direct API key
# ===================================================================
# For quick experiments you can skip virtual keys and pass the provider
# name along with the raw API key in the ``Authorization`` header.

print("\n" + "=" * 60)
print("Pattern 2: Provider slug + Authorization header")
print("=" * 60)

openai_direct = PortkeyModel(
    client_args={
        "api_key": PORTKEY_API_KEY,
        "provider": "openai",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    },
    model_id="gpt-4o-mini",
)

agent = Agent(
    model=openai_direct,
    system_prompt="You are a physics tutor. Be concise.",
)
print("[OpenAI via provider slug]")
print(agent(PROMPT))

anthropic_direct = PortkeyModel(
    client_args={
        "api_key": PORTKEY_API_KEY,
        "provider": "anthropic",
        "Authorization": f"Bearer {os.environ.get('ANTHROPIC_API_KEY')}",
    },
    model_id="claude-sonnet-4-20250514",
)

agent = Agent(
    model=anthropic_direct,
    system_prompt="You are a physics tutor. Be concise.",
)
print("\n[Anthropic via provider slug]")
print(agent(PROMPT))


# ===================================================================
# Pattern 3: Portkey config (fallbacks, load-balancing, etc.)
# ===================================================================
# A Portkey config defines advanced routing strategies.  You can create
# one in the dashboard and reference it by ID, or pass an inline dict.

print("\n" + "=" * 60)
print("Pattern 3a: Config ID from Portkey dashboard")
print("=" * 60)

# Using a config ID created in the Portkey dashboard.  The config
# specifies which providers, models, weights, and fallback order to use.
config_model = PortkeyModel(
    client_args={
        "api_key": PORTKEY_API_KEY,
        "config": os.environ.get("PORTKEY_CONFIG_ID"),
    },
    # When using a config, the model is typically defined inside the config
    # itself.  You still need to pass *some* model_id to satisfy the API
    # contract -- it may be overridden by the config.
    model_id="gpt-4o-mini",
)

agent = Agent(
    model=config_model,
    system_prompt="You are a physics tutor. Be concise.",
)
print("[Routed via Portkey config]")
print(agent(PROMPT))


print("\n" + "=" * 60)
print("Pattern 3b: Inline config with fallback")
print("=" * 60)

# You can also define the config inline as a Python dictionary.
# This example sets up a fallback: try OpenAI first, fall back to Anthropic.
inline_config = {
    "strategy": {"mode": "fallback"},
    "targets": [
        {
            "provider": "openai",
            "virtual_key": os.environ.get("PORTKEY_OPENAI_VIRTUAL_KEY"),
            "override_params": {"model": "gpt-4o-mini"},
        },
        {
            "provider": "anthropic",
            "virtual_key": os.environ.get("PORTKEY_ANTHROPIC_VIRTUAL_KEY"),
            "override_params": {"model": "claude-sonnet-4-20250514"},
        },
    ],
}

fallback_model = PortkeyModel(
    client_args={
        "api_key": PORTKEY_API_KEY,
        "config": inline_config,
    },
    model_id="gpt-4o-mini",
)

agent = Agent(
    model=fallback_model,
    system_prompt="You are a physics tutor. Be concise.",
)
print("[Routed via inline fallback config]")
print(agent(PROMPT))
