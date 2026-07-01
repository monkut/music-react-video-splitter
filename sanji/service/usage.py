"""Monthly stream usage counter — DynamoDB-backed, atomic increment (issue #5).

Table key: (user_id, period_key) where period_key = "YYYY-MM" (calendar month UTC).
Atomic ADD prevents double-counting under concurrent job submissions.
"""

import os
from datetime import UTC, datetime
from typing import Any

import boto3

USAGE_TABLE_ENV = "SANJI_USAGE_TABLE"
DEFAULT_USAGE_TABLE = "sanji-usage"


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
