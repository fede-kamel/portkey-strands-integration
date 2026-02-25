"""Integration tests: tool_choice strategies (auto, any, specific tool)."""

import pytest

from ._helpers import PROVIDER_CONFIGS, collect_stream, extract_text, proxy_model

pytestmark = pytest.mark.skipif(not PROVIDER_CONFIGS, reason="No API keys set")

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a city. Always use this when asked about weather.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name"}},
            "required": ["city"],
        }
    },
}

CALC_TOOL = {
    "name": "calculator",
    "description": "Evaluate arithmetic expressions.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }
    },
}


class TestToolChoice:
    @pytest.mark.asyncio
    async def test_tool_choice_any_forces_tool_call(self, provider_name):
        """tool_choice=any forces model to call at least one tool."""
        async with proxy_model(provider_name) as model:
            events = await collect_stream(
                model.stream(
                    [{"role": "user", "content": [{"text": "Hello, how are you?"}]}],
                    tool_specs=[WEATHER_TOOL, CALC_TOOL],
                    tool_choice={"any": {}},
                )
            )

        stop = [e for e in events if "messageStop" in e]
        assert stop[0]["messageStop"]["stopReason"] == "tool_use"

        tool_starts = [
            e
            for e in events
            if "contentBlockStart" in e and "toolUse" in e.get("contentBlockStart", {}).get("start", {})
        ]
        assert len(tool_starts) >= 1

    @pytest.mark.asyncio
    async def test_tool_choice_specific_tool_forces_named_tool(self, provider_name):
        """tool_choice={'tool': {'name': 'get_weather'}} forces that exact tool."""
        async with proxy_model(provider_name) as model:
            events = await collect_stream(
                model.stream(
                    [{"role": "user", "content": [{"text": "What time is it?"}]}],
                    tool_specs=[WEATHER_TOOL, CALC_TOOL],
                    tool_choice={"tool": {"name": "get_weather"}},
                )
            )

        tool_starts = [
            e
            for e in events
            if "contentBlockStart" in e and "toolUse" in e.get("contentBlockStart", {}).get("start", {})
        ]
        assert len(tool_starts) >= 1
        assert tool_starts[0]["contentBlockStart"]["start"]["toolUse"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_tool_choice_auto_allows_no_tool(self, provider_name):
        """tool_choice=auto allows model to skip tools on a factual question."""
        async with proxy_model(provider_name) as model:
            events = await collect_stream(
                model.stream(
                    [{"role": "user", "content": [{"text": "What is the capital of France?"}]}],
                    tool_specs=[WEATHER_TOOL, CALC_TOOL],
                    tool_choice={"auto": {}},
                )
            )

        stop = [e for e in events if "messageStop" in e]
        assert stop[0]["messageStop"]["stopReason"] == "end_turn"
        assert "paris" in extract_text(events).lower()
