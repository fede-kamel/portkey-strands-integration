"""Integration tests: structured output via Pydantic models.

Note: Only OpenAI is tested here. Anthropic's OpenAI-compat endpoint does not
support the beta.chat.completions.parse API required for structured output.
"""

import pytest
from pydantic import BaseModel

from ._helpers import PROVIDER_CONFIGS, collect_stream, proxy_model

pytestmark = pytest.mark.skipif(not PROVIDER_CONFIGS, reason="No API keys set")


class TestStructuredOutput:
    @pytest.mark.skipif("openai" not in PROVIDER_CONFIGS, reason="No OPENAI_API_KEY")
    @pytest.mark.asyncio
    async def test_structured_output_pydantic(self):
        """Structured output returns a validated Pydantic model."""

        class CityInfo(BaseModel):
            """Key facts about a city."""

            name: str
            country: str
            population_millions: float

        async with proxy_model("openai") as model:
            events = await collect_stream(
                model.structured_output(
                    CityInfo,
                    [{"role": "user", "content": [{"text": "Give me info about Tokyo."}]}],
                )
            )

        assert len(events) == 1
        output = events[0]["output"]
        assert isinstance(output, CityInfo)
        assert output.name.lower() == "tokyo"
        assert output.country.lower() == "japan"
        assert output.population_millions > 0
