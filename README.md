# strands-agents-portkey

[![CI — Lint & Type Check](https://github.com/fede-kamel/strands-portkey/actions/workflows/lint.yml/badge.svg)](https://github.com/fede-kamel/strands-portkey/actions/workflows/lint.yml)
[![CI — Unit Tests](https://github.com/fede-kamel/strands-portkey/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/fede-kamel/strands-portkey/actions/workflows/unit-tests.yml)
[![PyPI](https://img.shields.io/pypi/v/strands-agents-portkey)](https://pypi.org/project/strands-agents-portkey/)
[![Python](https://img.shields.io/pypi/pyversions/strands-agents-portkey)](https://pypi.org/project/strands-agents-portkey/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

A [Portkey](https://portkey.ai) model provider for the [Strands Agents SDK](https://github.com/strands-agents/sdk-python).

Route every Strands agent call through Portkey's AI gateway and instantly gain **fallbacks, load balancing, semantic caching, guardrails, and full observability** — without changing your agent logic.

---

## Why this matters

### Strands Agents — build AI agents simply

[Strands Agents](https://github.com/strands-agents/sdk-python) is an open-source Python SDK by AWS for building autonomous AI agents. Its model is elegantly simple:

```
Agent = Model + System Prompt + Tools
```

The SDK abstracts away every provider behind a single `Model` interface, so agent code stays clean and portable. Strands handles streaming, tool use, multi-turn conversation, parallel tool execution, and multi-agent coordination — you focus on the agent's behaviour, not the infrastructure.

### Portkey — production infrastructure for LLM applications

[Portkey](https://portkey.ai) is an AI gateway and observability platform purpose-built for production LLM applications. It sits between your application and 250+ LLM providers and delivers:

| Feature | What you get |
|---|---|
| **Fallbacks** | Automatic retry and failover across providers — agents never go down because one API is slow |
| **Load balancing** | Distribute requests across multiple API keys and models with configurable weights |
| **Semantic caching** | Cache similar requests, not just identical ones — dramatically cuts costs on repeated context |
| **Guardrails** | 60+ synchronous safety and compliance filters applied to every request and response |
| **Observability** | 40+ metrics per request: cost, latency, token counts, error rates — real-time dashboard |
| **Virtual keys** | Store provider API keys in an encrypted vault; rotate without redeploying agents |
| **Prompt management** | Version, test, and promote prompts across environments without code changes |

### Together — a complete production AI stack

Strands gives you the **application layer** (clean agent code, model-agnostic). Portkey gives you the **infrastructure layer** (reliability, visibility, cost control). Together:

```
Strands Agent
     │
     ▼
Portkey Gateway ──► OpenAI / Anthropic / Bedrock / Gemini / 250+ more
     │
     ▼
Observability Dashboard
Cost tracking · Fallbacks · Caching · Guardrails
```

You write agent code once. Portkey handles the rest in production.

---

## Installation

```bash
pip install strands-agents-portkey
```

---

## Quickstart

```python
from strands import Agent
from strands_portkey import PortkeyModel

model = PortkeyModel(
    model_id="gpt-4o",
    client_args={
        "api_key": "your-portkey-api-key",
        "virtual_key": "your-openai-virtual-key",
    },
)

agent = Agent(model=model)
response = agent("Summarise the key trends in AI infrastructure for 2025.")
print(response)
```

---

## Configuration

`PortkeyModel` accepts two sets of arguments:

### `model_id` and `params` — what to call

```python
model = PortkeyModel(
    model_id="claude-sonnet-4-6",   # any model supported by your provider
    params={                         # passed directly to the API
        "temperature": 0.7,
        "max_tokens": 1024,
    },
    client_args={...},
)
```

### `client_args` — how to route it

All arguments are forwarded to the underlying [`AsyncPortkey`](https://docs.portkey.ai/docs/api-reference/portkey-sdk-client) client:

```python
client_args={
    "api_key":      "pk-...",        # Portkey API key
    "virtual_key":  "vk-...",        # Provider key stored in Portkey vault
    "provider":     "anthropic",     # Provider slug (use with Authorization header)
    "Authorization": "Bearer sk-...", # Direct provider key (no virtual key)
    "config":       "pc-...",        # Portkey config ID (fallbacks, caching, etc.)
    "trace_id":     "req-abc123",    # Trace correlation ID
    "metadata":     {"user": "u1"},  # Custom metadata for observability
}
```

---

## Routing strategies

### Fallbacks

Automatically retry with a backup model if the primary fails:

```python
import json
from strands_portkey import PortkeyModel

fallback_config = {
    "strategy": {"mode": "fallback"},
    "targets": [
        {"virtual_key": "openai-vk",     "override_params": {"model": "gpt-4o"}},
        {"virtual_key": "anthropic-vk",  "override_params": {"model": "claude-sonnet-4-6"}},
        {"virtual_key": "bedrock-vk",    "override_params": {"model": "amazon.nova-pro-v1:0"}},
    ],
}

model = PortkeyModel(
    model_id="gpt-4o",
    client_args={
        "api_key": "pk-...",
        "config": json.dumps(fallback_config),
    },
)
```

### Load balancing

Distribute traffic across providers with weights:

```python
lb_config = {
    "strategy": {"mode": "loadbalance"},
    "targets": [
        {"virtual_key": "openai-key-1", "weight": 50},
        {"virtual_key": "openai-key-2", "weight": 30},
        {"virtual_key": "anthropic-vk", "weight": 20},
    ],
}
```

### Caching

Enable semantic caching to avoid redundant API calls:

```python
cache_config = {
    "cache": {"mode": "semantic", "max_age": 3600},
    "targets": [{"virtual_key": "openai-vk"}],
}
```

---

## Tool use

`PortkeyModel` fully supports Strands tool use, including parallel tool calls:

```python
from strands import Agent, tool
from strands_portkey import PortkeyModel

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22°C in {city}"

@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    return f"{city} has a population of 2.1 million"

model = PortkeyModel(
    model_id="gpt-4o",
    client_args={"api_key": "pk-...", "virtual_key": "openai-vk"},
)

agent = Agent(model=model, tools=[get_weather, get_population])
response = agent("What's the weather and population of Paris and London?")
```

The model can call both tools in parallel and combine the results in a single response.

---

## Structured output

```python
from pydantic import BaseModel
from strands_portkey import PortkeyModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

model = PortkeyModel(
    model_id="gpt-4o",
    client_args={"api_key": "pk-...", "virtual_key": "openai-vk"},
)

async for result in model.structured_output(Summary, [{"role": "user", "content": [{"text": "Summarise AI trends"}]}]):
    summary = result["output"]
    print(summary.title, summary.key_points)
```

---

## Observability

Every request routed through Portkey is automatically traced. View cost, latency, token usage, and errors in the [Portkey dashboard](https://app.portkey.ai).

Add metadata to correlate traces with your application:

```python
model = PortkeyModel(
    model_id="claude-sonnet-4-6",
    client_args={
        "api_key": "pk-...",
        "virtual_key": "anthropic-vk",
        "trace_id": "session-xyz",
        "metadata": {
            "agent": "research-agent",
            "user_id": "u-123",
            "environment": "production",
        },
    },
)
```

---

## Updating configuration at runtime

```python
model.update_config(model_id="gpt-4o-mini", params={"temperature": 0.3})
```

---

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow.

```bash
# Install hooks
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit

# Run all checks (lint + type check + unit tests)
hatch run check

# Run integration tests (requires API keys)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
hatch run test-integ
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Copyright (c) 2025 Federico Kamelhar and Portkey AI, Inc.

Third-party dependency licenses are documented in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
