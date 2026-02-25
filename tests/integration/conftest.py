"""Shared pytest fixtures for integration tests."""

import pytest

from ._helpers import PROVIDER_CONFIGS

PROVIDERS = [pytest.param(name, id=name) for name in PROVIDER_CONFIGS]


@pytest.fixture(params=PROVIDERS)
def provider_name(request):
    """Parametrize tests over all configured providers."""
    return request.param
