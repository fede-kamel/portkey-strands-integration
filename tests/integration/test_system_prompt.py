"""Integration tests: system prompt content blocks."""

import pytest

from ._helpers import PROVIDER_CONFIGS, collect_stream, extract_text, proxy_model

pytestmark = pytest.mark.skipif(not PROVIDER_CONFIGS, reason="No API keys set")


class TestSystemPromptContent:
    @pytest.mark.asyncio
    async def test_multi_block_system_prompt_influences_behavior(self, provider_name):
        """Multiple system content blocks are concatenated and respected."""
        async with proxy_model(provider_name) as model:
            events = await collect_stream(
                model.stream(
                    [{"role": "user", "content": [{"text": "What are you and what should you do?"}]}],
                    system_prompt_content=[
                        {"text": "You are a pirate assistant."},
                        {"text": "Always respond using pirate vocabulary."},
                    ],
                )
            )

        text = extract_text(events)
        pirate_words = ["arr", "matey", "ahoy", "pirate", "aye", "ye", "captain", "seas", "yarr"]
        assert any(w in text.lower() for w in pirate_words)

    @pytest.mark.asyncio
    async def test_cache_point_in_system_prompt_content_ignored(self, provider_name):
        """Cache point blocks in system_prompt_content are silently dropped."""
        async with proxy_model(provider_name) as model:
            events = await collect_stream(
                model.stream(
                    [{"role": "user", "content": [{"text": "Reply with just the word CONFIRMED."}]}],
                    system_prompt_content=[
                        {"text": "You are a helpful assistant. Always be brief."},
                        {"cachePoint": {"type": "default"}},
                        {"text": "Respond with exactly what the user requests."},
                    ],
                )
            )

        assert len(extract_text(events).strip()) > 0
        stop = [e for e in events if "messageStop" in e]
        assert stop[0]["messageStop"]["stopReason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_system_prompt_content_overrides_string(self, provider_name):
        """system_prompt_content takes precedence over system_prompt string."""
        async with proxy_model(provider_name) as model:
            events = await collect_stream(
                model.stream(
                    [{"role": "user", "content": [{"text": "What are you?"}]}],
                    system_prompt="You are a robot.",
                    system_prompt_content=[{"text": "You are a wizard. Use wizardly language."}],
                )
            )

        assert len(extract_text(events).strip()) > 0
        stop = [e for e in events if "messageStop" in e]
        assert stop[0]["messageStop"]["stopReason"] == "end_turn"
