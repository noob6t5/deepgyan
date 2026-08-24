import importlib

import pytest

from dashboard.backend import config


def test_validate_config_ok():
    config.validate_config()


def test_validate_config_missing_dirs(monkeypatch, tmp_path):
    missing_templates = tmp_path / "no-templates"
    missing_static = tmp_path / "no-static"

    monkeypatch.setattr(config, "TEMPLATES_DIR", str(missing_templates))
    monkeypatch.setattr(config, "STATIC_DIR", str(missing_static))

    with pytest.raises(RuntimeError):
        config.validate_config()


def test_openai_provider_uses_openai_key_not_sarvam_alias(monkeypatch):
    with monkeypatch.context() as env:
        env.setenv("INFERENCE_PROVIDER", "openai")
        env.setenv("INFERENCE_API_KEY", "")
        env.setenv("SARVAMAI_KEY", "sarvam-key")
        env.setenv("OPENAI_COMPAT_API_KEY", "compat-key")
        env.setenv("OPENAI_API_KEY", "openai-key")

        reloaded = importlib.reload(config)
        assert reloaded.INFERENCE_API_KEY == "openai-key"

    importlib.reload(config)


def test_openai_compatible_provider_uses_compatible_key(monkeypatch):
    with monkeypatch.context() as env:
        env.setenv("INFERENCE_PROVIDER", "openai-compatible")
        env.setenv("INFERENCE_API_KEY", "")
        env.setenv("SARVAMAI_KEY", "sarvam-key")
        env.setenv("OPENAI_COMPAT_API_KEY", "compat-key")
        env.setenv("OPENAI_API_KEY", "openai-key")

        reloaded = importlib.reload(config)
        assert reloaded.INFERENCE_API_KEY == "compat-key"

    importlib.reload(config)
