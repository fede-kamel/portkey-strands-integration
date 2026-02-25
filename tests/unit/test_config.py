"""Unit tests for strands_portkey._config (PortkeyConfig Pydantic model)."""

import pytest
from pydantic import ValidationError

from strands_portkey._config import PortkeyConfig


class TestPortkeyConfigValid:
    def test_minimal(self):
        cfg = PortkeyConfig(model_id="gpt-4o")
        assert cfg.model_id == "gpt-4o"
        assert cfg.params is None

    def test_with_params(self):
        cfg = PortkeyConfig(model_id="gpt-4o", params={"temperature": 0.5, "max_tokens": 512})
        assert cfg.params["temperature"] == 0.5
        assert cfg.params["max_tokens"] == 512

    def test_model_dump_roundtrip(self):
        cfg = PortkeyConfig(model_id="claude-sonnet-4-6", params={"top_p": 0.9})
        restored = PortkeyConfig(**cfg.model_dump())
        assert restored == cfg

    def test_model_dump_excludes_none_params(self):
        cfg = PortkeyConfig(model_id="gpt-4o")
        dumped = cfg.model_dump()
        assert dumped["model_id"] == "gpt-4o"
        assert dumped["params"] is None

    def test_equality(self):
        a = PortkeyConfig(model_id="gpt-4o", params={"temperature": 0.7})
        b = PortkeyConfig(model_id="gpt-4o", params={"temperature": 0.7})
        assert a == b

    def test_copy_update(self):
        cfg = PortkeyConfig(model_id="gpt-4o", params={"temperature": 0.7})
        updated = PortkeyConfig(**{**cfg.model_dump(), "model_id": "gpt-4o-mini"})
        assert updated.model_id == "gpt-4o-mini"
        assert updated.params == {"temperature": 0.7}


class TestPortkeyConfigValidation:
    def test_missing_model_id_raises(self):
        with pytest.raises(ValidationError, match="model_id"):
            PortkeyConfig()  # type: ignore[call-arg]

    def test_empty_model_id_raises(self):
        with pytest.raises(ValidationError, match="model_id must not be empty"):
            PortkeyConfig(model_id="")

    def test_whitespace_model_id_raises(self):
        with pytest.raises(ValidationError, match="model_id must not be empty"):
            PortkeyConfig(model_id="   ")

    def test_params_must_be_dict(self):
        with pytest.raises(ValidationError):
            PortkeyConfig(model_id="gpt-4o", params="bad")  # type: ignore[arg-type]

    def test_model_id_must_be_string(self):
        with pytest.raises(ValidationError):
            PortkeyConfig(model_id=42)  # type: ignore[arg-type]
