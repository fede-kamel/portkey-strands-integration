"""Getting structured Pydantic model output from a Portkey-backed agent.

Strands Agents can coerce model responses into a typed Pydantic schema.  This
is useful when you need machine-readable output -- e.g. for downstream
processing, database insertion, or API responses.

Two approaches are shown:

1. **Per-call structured output** -- pass ``structured_output_model`` when
   invoking the agent so only that call returns structured data.
2. **Agent-level default** -- set ``structured_output_model`` on the Agent
   constructor so *every* call returns structured data.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/structured_output.py
"""

import os

from pydantic import BaseModel, Field
from strands import Agent

from strands_portkey import PortkeyModel

# ---------------------------------------------------------------------------
# 1. Define Pydantic models for the desired output shapes
# ---------------------------------------------------------------------------


class MovieReview(BaseModel):
    """A structured movie review."""

    title: str = Field(description="The title of the movie")
    year: int = Field(description="The release year")
    rating: float = Field(description="Rating out of 10")
    genre: str = Field(description="Primary genre")
    summary: str = Field(description="A one-sentence summary of the review")
    pros: list[str] = Field(description="List of positive aspects")
    cons: list[str] = Field(description="List of negative aspects")


class CityInfo(BaseModel):
    """Key facts about a city."""

    name: str = Field(description="The city name")
    country: str = Field(description="The country the city is in")
    population: int = Field(description="Approximate population")
    known_for: list[str] = Field(description="Things the city is famous for")
    best_time_to_visit: str = Field(description="Best time of year to visit")


# ---------------------------------------------------------------------------
# 2. Create the Portkey model
# ---------------------------------------------------------------------------
model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    model_id="gpt-4o-mini",
)


# ===================================================================
# Approach 1: Per-call structured output
# ===================================================================
# Pass ``structured_output_model`` directly in the agent call.  This
# applies only to that single invocation.

print("=" * 60)
print("Approach 1: Per-call structured output")
print("=" * 60)

agent = Agent(
    model=model,
    system_prompt="You are a knowledgeable movie critic and travel guide.",
)

# Ask for a movie review -- the agent returns an AgentResult whose
# ``.structured_output`` attribute is a validated ``MovieReview`` instance.
result = agent(
    "Write a review of the movie Inception (2010).",
    structured_output_model=MovieReview,
)

review: MovieReview = result.structured_output  # type: ignore[assignment]
print(f"Title : {review.title}")
print(f"Year  : {review.year}")
print(f"Rating: {review.rating}/10")
print(f"Genre : {review.genre}")
print(f"Summary: {review.summary}")
print(f"Pros  : {review.pros}")
print(f"Cons  : {review.cons}")


# ===================================================================
# Approach 2: Agent-level default structured output
# ===================================================================
# When you set ``structured_output_model`` on the Agent constructor, every
# call automatically returns structured data of that type.

print("\n" + "=" * 60)
print("Approach 2: Agent-level default structured output")
print("=" * 60)

city_agent = Agent(
    model=model,
    system_prompt="You are a travel expert. Answer questions about cities.",
    structured_output_model=CityInfo,
)

for city_name in ["Tokyo", "Paris", "New York"]:
    result = city_agent(f"Tell me about {city_name}.")
    info: CityInfo = result.structured_output  # type: ignore[assignment]
    print(f"\n{info.name}, {info.country}")
    print(f"  Population: ~{info.population:,}")
    print(f"  Known for : {', '.join(info.known_for)}")
    print(f"  Best visit: {info.best_time_to_visit}")
