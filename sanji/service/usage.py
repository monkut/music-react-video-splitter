"""Stream usage counters — DynamoDB-backed, atomic increment (issues #5, #91).

Table key: (user_id, period_key) where period_key = "YYYY-MM" (calendar month UTC).
Atomic ADD prevents double-counting under concurrent job submissions.

The free tier also carries a lifetime ceiling (#91), stored in the same table
under the sentinel period key ``TOTAL``. Real period keys are always "YYYY-MM",
so the sentinel can never be produced by a month rollover — which is what lets
both counters share one table with no schema change.
"""

import os
from datetime import UTC, datetime
from typing import Any

import boto3

USAGE_TABLE_ENV = "SANJI_USAGE_TABLE"
DEFAULT_USAGE_TABLE = "sanji-usage"

# Sentinel period key holding the never-resetting lifetime run count (#91).
TOTAL_PERIOD_KEY = "TOTAL"


def _current_period() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


class UsageStore:
    """Owns per-user monthly stream counts in DynamoDB."""

    def __init__(self, table_name: str | None = None) -> None:
        self._table_name = table_name or os.getenv(USAGE_TABLE_ENV, DEFAULT_USAGE_TABLE)
        self._table = boto3.resource("dynamodb").Table(self._table_name)

    def get_monthly_count(self, user_id: str, period_key: str) -> int:
        response = self._table.get_item(
            Key={"user_id": user_id, "period_key": period_key}
        )
        item: dict[str, Any] | None = response.get("Item")
        return int(item["stream_count"]) if item else 0

    def increment_monthly_count(self, user_id: str, period_key: str) -> int:
        """Atomically increment the stream count; returns the new value."""
        response = self._table.update_item(
            Key={"user_id": user_id, "period_key": period_key},
            UpdateExpression="ADD stream_count :one",
            ExpressionAttributeValues={":one": 1},
            ReturnValues="UPDATED_NEW",
        )
        return int(response["Attributes"]["stream_count"])

    def get_current_count(self, user_id: str) -> int:
        return self.get_monthly_count(user_id, _current_period())

    def increment_current_count(self, user_id: str) -> int:
        return self.increment_monthly_count(user_id, _current_period())

    def get_total_count(self, user_id: str) -> int:
        """Lifetime run count across every period (#91)."""
        return self.get_monthly_count(user_id, TOTAL_PERIOD_KEY)

    def increment_total_count(self, user_id: str) -> int:
        """Atomically increment the lifetime count; returns the new value."""
        return self.increment_monthly_count(user_id, TOTAL_PERIOD_KEY)
