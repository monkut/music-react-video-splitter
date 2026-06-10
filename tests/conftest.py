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
