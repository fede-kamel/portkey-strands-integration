"""Integration tests: parallel tool calling."""

import pytest

from ._helpers import PROVIDER_CONFIGS, collect_stream, extract_text, proxy_model

pytestmark = pytest.mark.skipif(not PROVIDER_CONFIGS, reason="No API keys set")

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        }
    },
}


def extract_tool_calls(events: list) -> list[dict]:
    """Group stream events into per-tool-call dicts with id, name, and parsed input.

    Tool deltas carry no toolUseId — they are grouped by order between each
    contentBlockStart / contentBlockStop pair.
    """
    import json

    tool_calls = []
    current: dict | None = None

    for event in events:
        if "contentBlockStart" in event:
            start = event["contentBlockStart"]["start"]
            if "toolUse" in start:
                current = {
                    "toolUseId": start["toolUse"]["toolUseId"],
                    "name": start["toolUse"]["name"],
                    "input_raw": "",
                }
        elif "contentBlockDelta" in event and current is not None:
            delta = event["contentBlockDelta"]["delta"]
            if "toolUse" in delta:
                current["input_raw"] += delta["toolUse"].get("input", "")
        elif "contentBlockStop" in event and current is not None:
            try:
                current["input"] = json.loads(current["input_raw"]) if current["input_raw"] else {}
            except Exception:
                current["input"] = {}
            tool_calls.append(current)
            current = None

    return tool_calls


class TestParallelToolUse:
    @pytest.mark.asyncio
    async def test_parallel_tool_calls_emitted(self, provider_name):
        """Model emits multiple tool calls in a single response when asked about multiple cities."""
        prompt = (
            "Use get_weather to check the weather in Paris AND London at the same time. "
            "Call both tools in parallel, do not call them sequentially."
        )
        async with proxy_model(provider_name) as model:
            events = await collect_stream(
                model.stream(
                    [{"role": "user", "content": [{"text": prompt}]}],
                    tool_specs=[WEATHER_TOOL],
                )
            )

        stop = [e for e in events if "messageStop" in e]
        assert stop[0]["messageStop"]["stopReason"] == "tool_use"

        tool_calls = extract_tool_calls(events)
        assert len(tool_calls) >= 2, f"Expected at least 2 parallel tool calls, got {len(tool_calls)}"

        names = [tc["name"] for tc in tool_calls]
        assert all(name == "get_weather" for name in names)

        ids = [tc["toolUseId"] for tc in tool_calls]
        assert len(set(ids)) == len(ids), "Each parallel tool call must have a unique toolUseId"

        cities = {tc["input"].get("city", "").lower() for tc in tool_calls if tc.get("input")}
        assert len(cities) == 2, f"Expected calls for 2 distinct cities, got: {cities}"

    @pytest.mark.asyncio
    async def test_parallel_tool_results_round_trip(self, provider_name):
        """Full parallel tool use -> multiple results -> final answer cycle."""
        prompt = (
            "Use get_weather to check the weather in Tokyo AND Sydney at the same time. Call both tools in parallel."
        )
        async with proxy_model(provider_name) as model:
            events1 = await collect_stream(
                model.stream(
                    [{"role": "user", "content": [{"text": prompt}]}],
                    tool_specs=[WEATHER_TOOL],
                )
            )

        tool_calls = extract_tool_calls(events1)
        assert len(tool_calls) >= 2

        assistant_content = []
        assistant_text = extract_text(events1)
        if assistant_text.strip():
            assistant_content.append({"text": assistant_text})

        tool_results = []
        for tc in tool_calls:
            assistant_content.append(
                {"toolUse": {"toolUseId": tc["toolUseId"], "name": tc["name"], "input": tc["input"]}}
            )
            tool_results.append({"toolResult": {"toolUseId": tc["toolUseId"], "content": [{"text": "Sunny, 22°C"}]}})

        messages_turn2 = [
            {"role": "user", "content": [{"text": prompt}]},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": tool_results},
        ]

        async with proxy_model(provider_name) as model:
            events2 = await collect_stream(model.stream(messages_turn2, tool_specs=[WEATHER_TOOL]))

        assert len(extract_text(events2)) > 0

        stop2 = [e for e in events2 if "messageStop" in e]
        assert stop2[0]["messageStop"]["stopReason"] == "end_turn"
