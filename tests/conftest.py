"""Shared test fixtures for Portkey model provider tests."""

import pytest


@pytest.fixture
def portkey_client_args():
    """Default client args for testing."""
    return {
        "api_key": "test-portkey-api-key",
        "virtual_key": "test-virtual-key",
    }


@pytest.fixture
def portkey_model_config():
    """Default model config for testing."""
    return {
        "model_id": "gpt-4o",
        "params": {"temperature": 0.7, "max_tokens": 1000},
    }
