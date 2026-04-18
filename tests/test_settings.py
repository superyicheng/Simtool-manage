"""Tests for API-key resolution. Avoid hitting the real OS keyring."""

from __future__ import annotations

from unittest import mock

import pytest

from simtool import settings


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Make each test start with a clean environment."""

    monkeypatch.delenv(settings.ENV_VAR_ANTHROPIC, raising=False)


def test_env_var_wins_over_keyring(monkeypatch):
    monkeypatch.setenv(settings.ENV_VAR_ANTHROPIC, "sk-ant-from-env")

    with mock.patch.object(settings, "_keyring_key", return_value="sk-ant-from-keyring"):
        assert settings.get_anthropic_api_key() == "sk-ant-from-env"


def test_keyring_used_when_no_env():
    with mock.patch.object(settings, "_keyring_key", return_value="sk-ant-from-keyring"):
        with mock.patch.object(settings, "_DOTENV_AVAILABLE", False):
            assert settings.get_anthropic_api_key() == "sk-ant-from-keyring"


def test_require_raises_with_helpful_message():
    with mock.patch.object(settings, "_keyring_key", return_value=None):
        with mock.patch.object(settings, "_DOTENV_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc:
                settings.require_anthropic_api_key()
            msg = str(exc.value)
            assert "ANTHROPIC_API_KEY" in msg
            assert "simtool config set-key" in msg


def test_status_never_returns_full_key(monkeypatch):
    monkeypatch.setenv(settings.ENV_VAR_ANTHROPIC, "sk-ant-abcdefghijKLMNOP")
    s = settings.status()
    assert s["has_env_key"] is True
    # Only last 4 chars may be exposed
    assert s["effective_suffix"] == "MNOP"


def test_status_empty_when_nothing_configured():
    with mock.patch.object(settings, "_keyring_key", return_value=None):
        with mock.patch.object(settings, "_DOTENV_AVAILABLE", False):
            s = settings.status()
            assert s["has_env_key"] is False
            assert s["has_stored_key"] is False
            assert s["effective_suffix"] is None


def test_whitespace_only_env_treated_as_unset(monkeypatch):
    monkeypatch.setenv(settings.ENV_VAR_ANTHROPIC, "   ")
    with mock.patch.object(settings, "_keyring_key", return_value=None):
        with mock.patch.object(settings, "_DOTENV_AVAILABLE", False):
            assert settings.get_anthropic_api_key() is None
