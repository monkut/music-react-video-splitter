"""Stripe SDK configuration and startup validation."""

import os

_REQUIRED_STRIPE_ENV_VARS = (
    "STRIPE_SECRET_KEY",
    "STRIPE_WEBHOOK_SECRET",
    "STRIPE_PUBLISHABLE_KEY",
)


def validate_stripe_env() -> None:
    """Raise ValueError if any required Stripe env vars are absent."""
    missing = [var for var in _REQUIRED_STRIPE_ENV_VARS if not os.getenv(var)]
    if missing:
        raise ValueError(
            f"Missing required Stripe env var(s): {', '.join(missing)}"
        )
