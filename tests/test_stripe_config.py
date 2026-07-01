"""Unit tests for Stripe env var validation (issue #24)."""

import pytest

from sanji.settings import (
    STRIPE_PUBLISHABLE_KEY_ENV,
    STRIPE_SECRET_KEY_ENV,
    STRIPE_WEBHOOK_SECRET_ENV,
    validate_stripe_env_vars,
)

ALL_STRIPE_VARS = (STRIPE_SECRET_KEY_ENV, STRIPE_WEBHOOK_SECRET_ENV, STRIPE_PUBLISHABLE_KEY_ENV)


def test_validate_passes_when_all_vars_set(monkeypatch):
    for var in ALL_STRIPE_VARS:
        monkeypatch.setenv(var, "dummy")
    validate_stripe_env_vars()  # must not raise


@pytest.mark.parametrize("missing_var", ALL_STRIPE_VARS)
def test_validate_raises_when_one_var_missing(monkeypatch, missing_var):
    for var in ALL_STRIPE_VARS:
        monkeypatch.setenv(var, "dummy")
    monkeypatch.delenv(missing_var)
    with pytest.raises(ValueError, match=missing_var):
        validate_stripe_env_vars()


def test_validate_raises_when_all_vars_missing(monkeypatch):
    for var in ALL_STRIPE_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError):
        validate_stripe_env_vars()


def test_validate_error_message_names_all_missing_vars(monkeypatch):
    for var in ALL_STRIPE_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError) as exc_info:
        validate_stripe_env_vars()
    error_msg = str(exc_info.value)
    for var in ALL_STRIPE_VARS:
        assert var in error_msg
