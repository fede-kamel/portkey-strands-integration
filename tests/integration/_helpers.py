"""Shared helpers for integration tests (non-fixture utilities)."""

import base64
import os
from contextlib import asynccontextmanager

import httpx

from strands_portkey.model import PortkeyModel

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

PROVIDER_CONFIGS: dict = {}

if os.environ.get("OPENAI_API_KEY"):
    PROVIDER_CONFIGS["openai"] = {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.environ["OPENAI_API_KEY"],
        "model_id": os.environ.get("OPENAI_MODEL_ID", "gpt-4o"),
    }

if os.environ.get("ANTHROPIC_API_KEY"):
    PROVIDER_CONFIGS["anthropic"] = {
        "base_url": "https://api.anthropic.com/v1",
        "api_key": os.environ["ANTHROPIC_API_KEY"],
        "model_id": os.environ.get("ANTHROPIC_MODEL_ID", "claude-sonnet-4-6"),
    }


# ---------------------------------------------------------------------------
# Model proxy
# ---------------------------------------------------------------------------


@asynccontextmanager
async def proxy_model(provider_name: str, **params):
    """Create a PortkeyModel that proxies through to a real provider.

    Reconfigures the vendored OpenAI client inside AsyncPortkey to point at the
    real provider's API. The Portkey SDK code still runs (request building,
    header injection, response parsing) — just the destination changes.
    """
    config = PROVIDER_CONFIGS[provider_name]
    model = PortkeyModel(
        client_args={"api_key": "test-pk-key", "virtual_key": "test-vk"},
        model_id=config["model_id"],
        params=params,
    )

    oc = model.client.openai_client
    oc._base_url = httpx.URL(config["base_url"] + "/")
    oc.api_key = config["api_key"]
    oc._client._base_url = httpx.URL(config["base_url"] + "/")

    yield model


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------


async def collect_stream(async_gen):
    """Drain an async generator into a list of events."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


def extract_text(events):
    """Join all text deltas from a stream into a single string."""
    parts = []
    for e in events:
        if "contentBlockDelta" in e:
            delta = e["contentBlockDelta"]["delta"]
            if "text" in delta:
                parts.append(delta["text"])
    return "".join(parts)


def get_event_types(events):
    """Return the top-level key of each event (e.g. 'messageStart')."""
    return [list(e.keys())[0] for e in events]


# ---------------------------------------------------------------------------
# Shared test assets
# ---------------------------------------------------------------------------

# Minimal valid PNG: 1x1 red pixel
RED_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)
