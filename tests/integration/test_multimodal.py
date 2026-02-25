"""Integration tests: image (multimodal) input."""

import pytest

from ._helpers import PROVIDER_CONFIGS, RED_PIXEL_PNG, collect_stream, extract_text, proxy_model

pytestmark = pytest.mark.skipif(not PROVIDER_CONFIGS, reason="No API keys set")


class TestMultimodal:
    @pytest.mark.asyncio
    async def test_image_content_produces_response(self, provider_name):
        """Model receives an image and returns a non-empty description."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": "Describe this image briefly. What color is it?"},
                    {"image": {"format": "png", "source": {"bytes": RED_PIXEL_PNG}}},
                ],
            }
        ]
        async with proxy_model(provider_name) as model:
            events = await collect_stream(model.stream(messages))

        assert len(extract_text(events).strip()) > 0
        stop = [e for e in events if "messageStop" in e]
        assert stop[0]["messageStop"]["stopReason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_image_with_system_prompt(self, provider_name):
        """System prompt applies alongside image input."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": "What do you see?"},
                    {"image": {"format": "png", "source": {"bytes": RED_PIXEL_PNG}}},
                ],
            }
        ]
        async with proxy_model(provider_name) as model:
            events = await collect_stream(
                model.stream(
                    messages,
                    system_prompt="You are an image analysis assistant. Always mention the dominant color.",
                )
            )

        assert len(extract_text(events).strip()) > 0
        stop = [e for e in events if "messageStop" in e]
        assert stop[0]["messageStop"]["stopReason"] == "end_turn"
