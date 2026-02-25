"""Integration tests: model parameters (max_tokens, temperature)."""

import pytest

from ._helpers import PROVIDER_CONFIGS, collect_stream, extract_text, proxy_model

pytestmark = pytest.mark.skipif(not PROVIDER_CONFIGS, reason="No API keys set")


class TestModelParams:
    @pytest.mark.asyncio
    async def test_max_tokens_limit(self, provider_name):
        """max_tokens truncates response."""
        async with proxy_model(provider_name, max_tokens=5) as model:
            events = await collect_stream(
                model.stream([{"role": "user", "content": [{"text": "Write a long essay about computing history."}]}])
            )

        assert len(extract_text(events).split()) <= 20

    @pytest.mark.asyncio
    async def test_temperature_zero_deterministic(self, provider_name):
        """Temperature 0 produces consistent responses."""
        results = []
        for _ in range(2):
            async with proxy_model(provider_name, temperature=0, max_tokens=20) as model:
                events = await collect_stream(
                    model.stream([{"role": "user", "content": [{"text": "What is 2+2? Reply with just the number."}]}])
                )
                results.append(extract_text(events).strip())

        assert results[0] == results[1]
