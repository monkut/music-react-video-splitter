"""Tests for Stripe startup validation (issue #24)."""

import pytest

from sanji.service.stripe_config import validate_stripe_env

_ALL_VARS = ("STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET", "STRIPE_PUBLISHABLE_KEY")


@pytest.fixture(autouse=True)
def _clear_stripe_env(monkeypatch):
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)


def _set_all(monkeypatch, *, exclude: str | None = None) -> None:
    for var in _ALL_VARS:
        if var != exclude:
            monkeypatch.setenv(var, f"test_{var.lower()}")


def test_validate_passes_when_all_vars_set(monkeypatch):
    _set_all(monkeypatch)
    validate_stripe_env()  # must not raise


@pytest.mark.parametrize("missing", _ALL_VARS)
def test_validate_raises_when_var_missing(monkeypatch, missing):
    _set_all(monkeypatch, exclude=missing)
    with pytest.raises(ValueError, match=missing):
        validate_stripe_env()


def test_validate_names_all_missing_vars(monkeypatch):
    with pytest.raises(ValueError) as exc_info:
        validate_stripe_env()
    msg = str(exc_info.value)
    for var in _ALL_VARS:
        assert var in msg
