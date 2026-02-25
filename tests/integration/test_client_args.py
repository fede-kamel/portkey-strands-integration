"""Integration tests: Portkey client argument patterns."""

from strands_portkey.model import PortkeyModel


class TestPortkeyClientArgs:
    def test_virtual_key(self):
        """Virtual key pattern initializes correctly."""
        model = PortkeyModel(client_args={"api_key": "pk-test", "virtual_key": "vk-prod"}, model_id="gpt-4o")
        assert model.client is not None

    def test_provider_slug(self):
        """Provider slug pattern initializes correctly."""
        model = PortkeyModel(client_args={"api_key": "pk-test", "provider": "openai"}, model_id="gpt-4o")
        assert model.client is not None

    def test_config_id(self):
        """Config ID pattern initializes correctly."""
        model = PortkeyModel(client_args={"api_key": "pk-test", "config": "cfg-xxx"}, model_id="gpt-4o")
        assert model.client is not None

    def test_trace_and_metadata(self):
        """Trace ID and metadata pattern initializes correctly."""
        model = PortkeyModel(
            client_args={"api_key": "pk-test", "virtual_key": "vk", "trace_id": "t1", "metadata": {"env": "ci"}},
            model_id="gpt-4o",
        )
        assert model.client is not None
