"""DynamoDB-backed user store (D16: all-serverless, no RDS).

Single table keyed by ``user_id`` with a GSI on ``google_sub`` for OAuth lookup.
Subscription and usage records land in follow-up issues (#4, #5 enforcement).
"""

import os
import uuid
from datetime import UTC, datetime
from typing import Any

import boto3
from pydantic import BaseModel

from sanji.service.plans import DEFAULT_PLAN_CODE

USERS_TABLE_ENV = "SANJI_USERS_TABLE"
DEFAULT_USERS_TABLE = "sanji-users"
GOOGLE_SUB_INDEX = "google_sub-index"


class User(BaseModel):
    user_id: str
    google_sub: str
    email: str
    display_name: str
    avatar_url: str | None = None
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None
    current_plan_code: str = DEFAULT_PLAN_CODE
    created_at: str
    updated_at: str


class UserStore:
    """Owns User persistence in DynamoDB."""

    def __init__(self, table_name: str | None = None) -> None:
        self._table_name = table_name or os.getenv(USERS_TABLE_ENV, DEFAULT_USERS_TABLE)
        self._table = boto3.resource("dynamodb").Table(self._table_name)

    def create(
        self,
        *,
        google_sub: str,
        email: str,
        display_name: str,
        avatar_url: str | None = None,
    ) -> User:
        now = datetime.now(UTC).isoformat()
        user = User(
            user_id=str(uuid.uuid4()),
            google_sub=google_sub,
            email=email,
            display_name=display_name,
            avatar_url=avatar_url,
            created_at=now,
            updated_at=now,
        )
        self._table.put_item(Item=user.model_dump(exclude_none=True))
        return user

    def get(self, user_id: str) -> User | None:
        response = self._table.get_item(Key={"user_id": user_id})
        item: dict[str, Any] | None = response.get("Item")
        return User.model_validate(item) if item else None

    def update_stripe_customer(self, user_id: str, *, stripe_customer_id: str) -> None:
        """Persist Stripe Customer ID for the user."""
        now = datetime.now(UTC).isoformat()
        self._table.update_item(
            Key={"user_id": user_id},
            UpdateExpression="SET stripe_customer_id = :cid, updated_at = :now",
            ExpressionAttributeValues={":cid": stripe_customer_id, ":now": now},
        )

    def get_by_google_sub(self, google_sub: str) -> User | None:
        response = self._table.query(
            IndexName=GOOGLE_SUB_INDEX,
            KeyConditionExpression="google_sub = :sub",
            ExpressionAttributeValues={":sub": google_sub},
            Limit=1,
        )
        items = response.get("Items", [])
        return User.model_validate(items[0]) if items else None
