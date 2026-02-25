"""Integration tests: basic text streaming, system prompts, multi-turn."""

import pytest

from ._helpers import PROVIDER_CONFIGS, collect_stream, extract_text, get_event_types, proxy_model

pytestmark = pytest.mark.skipif(not PROVIDER_CONFIGS, reason="No API keys set")


class TestBasicStreaming:
    @pytest.mark.asyncio
    async def test_simple_text_response(self, provider_name):
        """Full pipeline with real LLM: text streaming works."""
        async with proxy_model(provider_name) as model:
            events = await collect_stream(
                model.stream([{"role": "user", "content": [{"text": "Say 'hello world' and nothing else."}]}])
            )

        types = get_event_types(events)
        assert types[0] == "messageStart"
        assert "contentBlockStart" in types
        assert "contentBlockDelta" in types
        assert "contentBlockStop" in types
        assert "messageStop" in types
        assert "hello" in extract_text(events).lower()

    @pytest.mark.asyncio
    async def test_stream_event_order(self, provider_name):
        """Events come in correct order."""
        async with proxy_model(provider_name) as model:
            events = await collect_stream(model.stream([{"role": "user", "content": [{"text": "Say 'ok'"}]}]))

        types = get_event_types(events)
        assert types.index("messageStart") < types.index("contentBlockStart")
        assert types.index("contentBlockStart") < types.index("contentBlockStop")
        assert types.index("contentBlockStop") < types.index("messageStop")

    @pytest.mark.asyncio
    async def test_usage_metadata(self, provider_name):
        """Token usage is reported from real API."""
        async with proxy_model(provider_name) as model:
            events = await collect_stream(model.stream([{"role": "user", "content": [{"text": "Say 'hi'"}]}]))

        meta = [e for e in events if "metadata" in e]
        assert len(meta) == 1
        usage = meta[0]["metadata"]["usage"]
        assert usage["inputTokens"] > 0
        assert usage["outputTokens"] > 0

    @pytest.mark.asyncio
    async def test_end_turn_stop_reason(self, provider_name):
        """Normal completion has end_turn stop reason."""
        async with proxy_model(provider_name) as model:
            events = await collect_stream(model.stream([{"role": "user", "content": [{"text": "Say 'done'"}]}]))

        stop = [e for e in events if "messageStop" in e]
        assert stop[0]["messageStop"]["stopReason"] == "end_turn"


class TestSystemPrompts:
    @pytest.mark.asyncio
    async def test_system_prompt_affects_behavior(self, provider_name):
        """System prompt changes model behavior."""
        async with proxy_model(provider_name) as model:
            events = await collect_stream(
                model.stream(
                    [{"role": "user", "content": [{"text": "What are you?"}]}],
                    system_prompt="You are a pirate. Always respond in pirate speak.",
                )
            )

        text = extract_text(events)
        pirate_words = ["arr", "matey", "ahoy", "pirate", "aye", "ye", "captain", "ship", "seas", "yarr"]
        assert any(w in text.lower() for w in pirate_words)


class TestMultiTurn:
    @pytest.mark.asyncio
    async def test_multi_turn_context(self, provider_name):
        """Model maintains context across turns."""
        async with proxy_model(provider_name) as model:
            messages = [
                {"role": "user", "content": [{"text": "My name is Zephyr. Remember that."}]},
                {"role": "assistant", "content": [{"text": "Got it! Your name is Zephyr."}]},
                {"role": "user", "content": [{"text": "What is my name?"}]},
            ]
            events = await collect_stream(model.stream(messages))

        assert "zephyr" in extract_text(events).lower()
