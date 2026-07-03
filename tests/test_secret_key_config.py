"""Unit tests for SECRET_KEY env var validation (issue #31)."""

import pytest

from sanji.settings import SECRET_KEY_ENV, validate_secret_key_env_var


def test_validate_returns_value_when_set(monkeypatch):
    monkeypatch.setenv(SECRET_KEY_ENV, "a-long-random-secret")
    assert validate_secret_key_env_var() == "a-long-random-secret"


def test_validate_raises_when_missing(monkeypatch):
    monkeypatch.delenv(SECRET_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=SECRET_KEY_ENV):
        validate_secret_key_env_var()


def test_validate_raises_when_empty(monkeypatch):
    monkeypatch.setenv(SECRET_KEY_ENV, "")
    with pytest.raises(ValueError, match=SECRET_KEY_ENV):
        validate_secret_key_env_var()


def test_create_app_fails_fast_without_secret_key(monkeypatch):
    """create_app must not fall back to a hardcoded session-signing key."""
    monkeypatch.delenv(SECRET_KEY_ENV, raising=False)
    from sanji.service.app import create_app

    with pytest.raises(ValueError, match=SECRET_KEY_ENV):
        create_app()
