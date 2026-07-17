"""Tests for GET /me/usage — usage count, plan limit, subscription status (#83).

The endpoint surfaces to the client the same usage count + plan limit the
``POST /jobs`` cap check enforces, plus the webhook-synced subscription status.
"""

from datetime import UTC, datetime

import boto3
import pytest
from moto import mock_aws

from sanji.service.app import create_app
from sanji.service.billing import (
    DEFAULT_SUBSCRIPTIONS_TABLE,
    SubscriptionRecord,
    SubscriptionStore,
)
from sanji.service.plans import get_plan
from sanji.service.usage import DEFAULT_USAGE_TABLE, UsageStore
from sanji.service.users import GOOGLE_SUB_INDEX, UserStore


@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")


def _create_tables(dynamodb) -> None:
    dynamodb.create_table(
        TableName="sanji-users",
        AttributeDefinitions=[
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "google_sub", "AttributeType": "S"},
        ],
        KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
        GlobalSecondaryIndexes=[
            {
                "IndexName": GOOGLE_SUB_INDEX,
                "KeySchema": [{"AttributeName": "google_sub", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    dynamodb.create_table(
        TableName=DEFAULT_USAGE_TABLE,
        AttributeDefinitions=[
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "period_key", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "user_id", "KeyType": "HASH"},
            {"AttributeName": "period_key", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    dynamodb.create_table(
        TableName=DEFAULT_SUBSCRIPTIONS_TABLE,
        AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
        BillingMode="PAY_PER_REQUEST",
    )


@pytest.fixture
def authed():
    with mock_aws():
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
        _create_tables(dynamodb)
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(QueueName="sanji-jobs-usage-test")["QueueUrl"]
        app = create_app(config_overrides={"SQS_QUEUE_URL": queue_url})
        app.config["TESTING"] = True
        user = UserStore().create(
            google_sub="g-usage",
            email="usage@example.com",
            display_name="Usage User",
        )
        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["user_id"] = user.user_id
            yield client, user


@pytest.fixture
def anon_client():
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(QueueName="sanji-jobs-anon-test")["QueueUrl"]
        app = create_app(config_overrides={"SQS_QUEUE_URL": queue_url})
        app.config["TESTING"] = True
        yield app.test_client()


def test_usage_requires_login(anon_client) -> None:
    assert anon_client.get("/me/usage").status_code == 401


def test_usage_reports_count_and_plan_limit(authed) -> None:
    client, user = authed
    UsageStore().increment_current_count(user.user_id)
    UsageStore().increment_current_count(user.user_id)

    resp = client.get("/me/usage")
    assert resp.status_code == 200
    body = resp.json
    assert body["period"] == datetime.now(UTC).strftime("%Y-%m")
    assert body["plan_code"] == "free"
    assert body["usage"]["stream_count"] == 2
    free_plan = get_plan("free")
    assert free_plan is not None
    assert body["usage"]["stream_limit"] == free_plan.monthly_stream_limit


def test_usage_subscription_absent_is_null(authed) -> None:
    client, _user = authed
    resp = client.get("/me/usage")
    assert resp.status_code == 200
    assert resp.json["subscription"] is None


def test_usage_reports_active_subscription(authed) -> None:
    client, user = authed
    SubscriptionStore().put(
        SubscriptionRecord(
            user_id=user.user_id,
            stripe_subscription_id="sub_123",
            stripe_customer_id="cus_123",
            plan_code="pro",
            status="active",
            current_period_end="2026-08-01T00:00:00Z",
            updated_at="2026-07-17T00:00:00Z",
        )
    )

    resp = client.get("/me/usage")
    assert resp.status_code == 200
    subscription = resp.json["subscription"]
    assert subscription["status"] == "active"
    assert subscription["plan_code"] == "pro"
    assert subscription["current_period_end"] == "2026-08-01T00:00:00Z"
