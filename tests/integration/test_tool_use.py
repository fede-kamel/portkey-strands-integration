"""Integration tests: tool use and tool result round-trips."""

import json

import pytest

from ._helpers import PROVIDER_CONFIGS, collect_stream, extract_text, proxy_model

pytestmark = pytest.mark.skipif(not PROVIDER_CONFIGS, reason="No API keys set")

CALC_TOOL = {
    "name": "calculator",
    "description": "Performs arithmetic calculations. Use this for any math.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The math expression to evaluate"},
            },
            "required": ["expression"],
        }
    },
}


class TestToolUse:
    @pytest.mark.asyncio
    async def test_model_calls_tool(self, provider_name):
        """Model uses a tool and emits proper events."""
        prompt = "Use the calculator tool to compute 15 * 37. You must use the tool."
        async with proxy_model(provider_name) as model:
            events = await collect_stream(
                model.stream(
                    [{"role": "user", "content": [{"text": prompt}]}],
                    tool_specs=[CALC_TOOL],
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
        assert tool_starts[0]["contentBlockStart"]["start"]["toolUse"]["name"] == "calculator"
        assert tool_starts[0]["contentBlockStart"]["start"]["toolUse"]["toolUseId"]

        tool_deltas = [
            e
            for e in events
            if "contentBlockDelta" in e and "toolUse" in e.get("contentBlockDelta", {}).get("delta", {})
        ]
        combined = "".join(d["contentBlockDelta"]["delta"]["toolUse"]["input"] for d in tool_deltas)
        assert "expression" in json.loads(combined)

    @pytest.mark.asyncio
    async def test_tool_result_round_trip(self, provider_name):
        """Full tool use -> result -> final answer cycle."""
        prompt = "Use the calculator to compute 15 * 37. You must use the calculator tool."
        async with proxy_model(provider_name) as model:
            events1 = await collect_stream(
                model.stream(
                    [{"role": "user", "content": [{"text": prompt}]}],
                    tool_specs=[CALC_TOOL],
                )
            )

        tool_starts = [
            e
            for e in events1
            if "contentBlockStart" in e and "toolUse" in e.get("contentBlockStart", {}).get("start", {})
        ]
        tool_use_id = tool_starts[0]["contentBlockStart"]["start"]["toolUse"]["toolUseId"]

        tool_deltas = [
            e
            for e in events1
            if "contentBlockDelta" in e and "toolUse" in e.get("contentBlockDelta", {}).get("delta", {})
        ]
        tool_input = json.loads("".join(d["contentBlockDelta"]["delta"]["toolUse"]["input"] for d in tool_deltas))

        text_parts = [
            e for e in events1 if "contentBlockDelta" in e and "text" in e.get("contentBlockDelta", {}).get("delta", {})
        ]
        assistant_text = "".join(d["contentBlockDelta"]["delta"]["text"] for d in text_parts)

        assistant_content = []
        if assistant_text.strip():
            assistant_content.append({"text": assistant_text})
        assistant_content.append({"toolUse": {"toolUseId": tool_use_id, "name": "calculator", "input": tool_input}})

        messages_turn2 = [
            {"role": "user", "content": [{"text": prompt}]},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": [{"toolResult": {"toolUseId": tool_use_id, "content": [{"text": "555"}]}}]},
        ]

        async with proxy_model(provider_name) as model:
            events2 = await collect_stream(model.stream(messages_turn2, tool_specs=[CALC_TOOL]))

        assert "555" in extract_text(events2)
        stop2 = [e for e in events2 if "messageStop" in e]
        assert stop2[0]["messageStop"]["stopReason"] == "end_turn"
