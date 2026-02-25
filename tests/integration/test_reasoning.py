"""Integration tests: reasoning/thinking content events.

Real reasoning models (OpenAI o1/o3, Anthropic extended thinking) are expensive
and slow. Instead we verify the routing behavior by injecting a synthetic
reasoning_content delta via a mock at the API boundary. This exercises the full
stream() dispatch path on a real PortkeyModel instance while keeping tests fast.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._helpers import PROVIDER_CONFIGS, collect_stream, extract_text, proxy_model

pytestmark = pytest.mark.skipif(not PROVIDER_CONFIGS, reason="No API keys set")


def _make_reasoning_chunk(reasoning_text):
    """Build a mock chunk with a reasoning_content delta."""
    delta = MagicMock()
    delta.content = None
    delta.tool_calls = None
    delta.reasoning_content = reasoning_text
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _make_text_chunk(text, finish_reason=None):
    """Build a mock chunk with a text delta (no reasoning_content)."""
    delta = MagicMock()
    delta.content = text
    delta.tool_calls = None
    del delta.reasoning_content  # hasattr returns False
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _make_usage_chunk():
    """Build a mock chunk carrying token usage."""
    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 10
    usage.total_tokens = 15
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = usage
    return chunk


class TestReasoningContent:
    @pytest.mark.asyncio
    async def test_reasoning_delta_yields_reasoning_content_event(self, provider_name):
        """reasoning_content deltas are forwarded as reasoningContent stream events."""

        async def mock_stream():
            yield _make_reasoning_chunk("Let me think about this carefully.")
            yield _make_text_chunk("The answer is 42.", finish_reason="stop")
            yield _make_usage_chunk()

        async with proxy_model(provider_name) as model:
            with patch.object(model.client.chat.completions, "create", new=AsyncMock(return_value=mock_stream())):
                events = await collect_stream(
                    model.stream([{"role": "user", "content": [{"text": "Think about 6*7"}]}])
                )

        reasoning_events = [
            e
            for e in events
            if "contentBlockDelta" in e and "reasoningContent" in e.get("contentBlockDelta", {}).get("delta", {})
        ]
        assert len(reasoning_events) == 1
        assert (
            reasoning_events[0]["contentBlockDelta"]["delta"]["reasoningContent"]["text"]
            == "Let me think about this carefully."
        )
        assert "42" in extract_text(events)
