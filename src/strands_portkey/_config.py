"""Pydantic configuration model for the Portkey model provider."""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class PortkeyConfig(BaseModel):
    """Validated configuration for PortkeyModel.

    Attributes:
        model_id: Model ID (e.g., "gpt-4o", "claude-sonnet-4-6").
        params: Additional model parameters forwarded to the API
            (e.g., ``{"max_tokens": 1024, "temperature": 0.7}``).
    """

    model_id: str = Field(..., description="Model ID, e.g. 'gpt-4o'")
    params: Optional[dict[str, Any]] = Field(None, description="Extra model parameters")

    @field_validator("model_id")
    @classmethod
    def model_id_not_empty(cls, v: str) -> str:
        """Reject blank model IDs."""
        if not v.strip():
            raise ValueError("model_id must not be empty or whitespace")
        return v
