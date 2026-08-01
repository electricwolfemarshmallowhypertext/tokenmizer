from __future__ import annotations

import pytest

from tokenmizer.config.settings import Settings


def test_environment_overrides_yaml(tmp_path, monkeypatch):
    config = tmp_path / "tokenmizer.yaml"
    config.write_text("provider: anthropic\nstate_backend: memory\n", encoding="utf-8")
    monkeypatch.setenv("TOKENMIZER_PROVIDER", "openai")
    monkeypatch.setenv("TOKENMIZER_STATE_BACKEND", "redis")

    settings = Settings.from_yaml(str(config))

    assert settings.provider == "openai"
    assert settings.state_backend == "redis"


def test_empty_environment_value_overrides_yaml_where_valid(tmp_path, monkeypatch):
    config = tmp_path / "tokenmizer.yaml"
    config.write_text("api_key: yaml-secret\n", encoding="utf-8")
    monkeypatch.setenv("TOKENMIZER_API_KEY", "")

    assert Settings.from_yaml(str(config)).api_key == ""


def test_required_empty_environment_value_is_rejected(tmp_path, monkeypatch):
    class RequiredSettings(Settings):
        required_marker: str

    config = tmp_path / "tokenmizer.yaml"
    config.write_text("required_marker: from-yaml\n", encoding="utf-8")
    monkeypatch.setenv("TOKENMIZER_REQUIRED_MARKER", "")

    with pytest.raises(ValueError, match="cannot be empty"):
        RequiredSettings.from_yaml(str(config))


def test_yaml_overrides_defaults_when_environment_is_absent(tmp_path, monkeypatch):
    config = tmp_path / "tokenmizer.yaml"
    config.write_text("provider: deepseek\n", encoding="utf-8")
    monkeypatch.delenv("TOKENMIZER_PROVIDER", raising=False)

    assert Settings.from_yaml(str(config)).provider == "deepseek"
