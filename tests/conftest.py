"""Test environment bootstrap.

sandjig resolves AWS_ACCOUNT_ID via STS at import time when the env var is
unset (sandjig/settings.py) — these must be in place before any test module
imports the service code, hence module-level here rather than a fixture.
"""

import os

os.environ.setdefault("AWS_ACCOUNT_ID", "000000000000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

# Stripe — dummy values so create_app() passes fail-fast validation in tests
os.environ.setdefault("STRIPE_SECRET_KEY", "sk_test_dummy")
os.environ.setdefault("STRIPE_WEBHOOK_SECRET", "whsec_dummy")
os.environ.setdefault("STRIPE_PUBLISHABLE_KEY", "pk_test_dummy")
