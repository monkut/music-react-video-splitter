"""Shared AWS client access.

The single place a boto3 S3 client is constructed: each client construction
rebuilds the session and credential chain, which is measurable latency (#40),
so callers share one memoized instance instead of building their own.
"""

from functools import cache
from typing import Any

import boto3


@cache
def get_s3_client() -> Any:
    return boto3.client("s3")
