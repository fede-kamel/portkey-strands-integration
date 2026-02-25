"""Integration tests: error translation (context overflow, throttling)."""

from unittest.mock import AsyncMock, patch

import pytest
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

from ._helpers import PROVIDER_CONFIGS, collect_stream, proxy_model

pytestmark = pytest.mark.skipif(not PROVIDER_CONFIGS, reason="No API keys set")


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_context_overflow_is_mapped(self, provider_name):
        """ContextWindowOverflowException is raised for context overflow errors."""
        async with proxy_model(provider_name) as model:
            with patch.object(
                model.client.chat.completions,
                "create",
                new=AsyncMock(side_effect=Exception("maximum context length exceeded")),
            ):
                with pytest.raises(ContextWindowOverflowException):
                    await collect_stream(model.stream([{"role": "user", "content": [{"text": "Hi"}]}]))

    @pytest.mark.asyncio
    async def test_throttle_is_mapped(self, provider_name):
        """ModelThrottledException is raised for rate-limit errors."""
        async with proxy_model(provider_name) as model:
            with patch.object(
                model.client.chat.completions,
                "create",
                new=AsyncMock(side_effect=Exception("429 rate limit exceeded")),
            ):
                with pytest.raises(ModelThrottledException):
                    await collect_stream(model.stream([{"role": "user", "content": [{"text": "Hi"}]}]))
