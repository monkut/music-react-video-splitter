"""Tests for the Stripe billing integration (issue #4).

AWS (DynamoDB, SQS) is mocked with moto; the Stripe SDK is patched so tests
run offline without real credentials.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import boto3
import pytest
import stripe
from moto import mock_aws

from sanji.service.billing import (
    DEFAULT_SUBSCRIPTIONS_TABLE,
    DEFAULT_WEBHOOK_EVENTS_TABLE,
    ProcessedEventStore,
    SubscriptionStore,
)
from sanji.service.plans import PLANS
from sanji.service.users import GOOGLE_SUB_INDEX, UserStore

PRO_PRICE_ID = "price_test_pro"


@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_testing")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_testing")
    monkeypatch.setenv("STRIPE_PUBLISHABLE_KEY", "pk_test_testing")


@pytest.fixture
def pro_price_id(monkeypatch):
    pro_plan = next(plan for plan in PLANS if plan.code == "pro")
    monkeypatch.setattr(pro_plan, "stripe_price_id", PRO_PRICE_ID)
    return PRO_PRICE_ID


def _create_tables():
    dynamodb = boto3.client("dynamodb", region_name="us-west-2")
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
        TableName=DEFAULT_SUBSCRIPTIONS_TABLE,
        AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
        BillingMode="PAY_PER_REQUEST",
    )
    dynamodb.create_table(
        TableName=DEFAULT_WEBHOOK_EVENTS_TABLE,
        AttributeDefinitions=[
            {"AttributeName": "stripe_event_id", "AttributeType": "S"}
        ],
        KeySchema=[{"AttributeName": "stripe_event_id", "KeyType": "HASH"}],
        BillingMode="PAY_PER_REQUEST",
    )


@pytest.fixture
def app_client():
    with mock_aws():
        _create_tables()
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(QueueName="sanji-jobs-test")["QueueUrl"]

        from sanji.service.app import create_app

        app = create_app(config_overrides={"SQS_QUEUE_URL": queue_url})
        app.config["TESTING"] = True
        app.secret_key = "test-secret"
        yield app.test_client()


def _seed_user(client):
    store = UserStore()
    user = store.create(
        google_sub="google-sub-billing",
        email="billing@example.com",
        display_name="Billing User",
    )
    with client.session_transaction() as sess:
        sess["user_id"] = user.user_id
    return user


def _subscription_payload(user_id, *, status="active", price_id=PRO_PRICE_ID):
    """Post-basil (2025-03-31) Stripe shape: current_period_end lives on items."""
    return {
        "id": "sub_test_1",
        "customer": "cus_test_1",
        "status": status,
        "metadata": {"user_id": user_id},
        "items": {
            "data": [{"price": {"id": price_id}, "current_period_end": 1782000000}]
        },
    }


def _legacy_subscription_payload(user_id, *, status="active", price_id=PRO_PRICE_ID):
    """Pre-basil shape: current_period_end at the subscription top level."""
    return {
        "id": "sub_test_1",
        "customer": "cus_test_1",
        "status": status,
        "current_period_end": 1782000000,
        "metadata": {"user_id": user_id},
        "items": {"data": [{"price": {"id": price_id}}]},
    }


def _subscription_updated_event(event_id="evt_test_1"):
    return {
        "id": event_id,
        "type": "customer.subscription.updated",
        "data": {"object": {"id": "sub_test_1"}},
    }


def test_checkout_creates_stripe_session(app_client, pro_price_id):
    user = _seed_user(app_client)
    fake_session = MagicMock(url="https://checkout.stripe.com/c/pay/test")

    with (
        patch(
            "stripe.Customer.create", return_value=MagicMock(id="cus_test_1")
        ) as customer_create,
        patch(
            "stripe.checkout.Session.create", return_value=fake_session
        ) as session_create,
    ):
        response = app_client.post("/billing/checkout", json={"price_id": pro_price_id})

    assert response.status_code == 200, response.data
    assert response.json["url"] == fake_session.url
    customer_create.assert_called_once()

    session_kwargs = session_create.call_args.kwargs
    assert session_kwargs["mode"] == "subscription"
    assert session_kwargs["client_reference_id"] == user.user_id
    assert (
        session_kwargs["idempotency_key"] == f"checkout-{user.user_id}-{pro_price_id}"
    )

    stored = UserStore().get(user.user_id)
    assert stored is not None
    assert stored.stripe_customer_id == "cus_test_1"


def test_webhook_valid_signature_updates_subscription(app_client, pro_price_id):
    user = _seed_user(app_client)

    with (
        patch(
            "stripe.Webhook.construct_event",
            return_value=_subscription_updated_event(),
        ),
        patch(
            "stripe.Subscription.retrieve",
            return_value=_subscription_payload(user.user_id),
        ),
    ):
        response = app_client.post(
            "/webhooks/stripe",
            data=b"{}",
            headers={"Stripe-Signature": "t=1,v1=testsig"},
        )

    assert response.status_code == 200, response.data
    record = SubscriptionStore().get(user.user_id)
    assert record is not None
    assert record.stripe_subscription_id == "sub_test_1"


def test_webhook_syncs_period_end_from_basil_items_shape(app_client, pro_price_id):
    """Post-basil payloads carry current_period_end on items.data[] (#36)."""
    user = _seed_user(app_client)
    with (
        patch(
            "stripe.Webhook.construct_event",
            return_value=_subscription_updated_event(),
        ),
        patch(
            "stripe.Subscription.retrieve",
            return_value=_subscription_payload(user.user_id),
        ),
    ):
        response = app_client.post(
            "/webhooks/stripe",
            data=b"{}",
            headers={"Stripe-Signature": "t=1,v1=testsig"},
        )

    assert response.status_code == 200, response.data
    record = SubscriptionStore().get(user.user_id)
    assert record is not None
    expected = datetime.fromtimestamp(1782000000, tz=UTC).isoformat()
    assert record.current_period_end == expected


def test_webhook_syncs_period_end_from_legacy_top_level_field(app_client, pro_price_id):
    """Pre-basil payloads still sync via the top-level fallback (#36)."""
    user = _seed_user(app_client)
    with (
        patch(
            "stripe.Webhook.construct_event",
            return_value=_subscription_updated_event(),
        ),
        patch(
            "stripe.Subscription.retrieve",
            return_value=_legacy_subscription_payload(user.user_id),
        ),
    ):
        response = app_client.post(
            "/webhooks/stripe",
            data=b"{}",
            headers={"Stripe-Signature": "t=1,v1=testsig"},
        )

    assert response.status_code == 200, response.data
    record = SubscriptionStore().get(user.user_id)
    assert record is not None
    expected = datetime.fromtimestamp(1782000000, tz=UTC).isoformat()
    assert record.current_period_end == expected
    assert record.stripe_customer_id == "cus_test_1"
    assert record.plan_code == "pro"
    assert record.status == "active"
    assert ProcessedEventStore().is_processed("evt_test_1")


def test_webhook_invalid_signature_returns_400(app_client):
    with patch(
        "stripe.Webhook.construct_event",
        side_effect=stripe.SignatureVerificationError("bad signature", "sig-header"),
    ):
        response = app_client.post(
            "/webhooks/stripe",
            data=b"{}",
            headers={"Stripe-Signature": "bad"},
        )

    assert response.status_code == 400
    assert response.json["error"] == "invalid_signature"


def test_duplicate_webhook_event_is_idempotent(app_client, pro_price_id):
    user = _seed_user(app_client)

    with (
        patch(
            "stripe.Webhook.construct_event",
            return_value=_subscription_updated_event(),
        ),
        patch(
            "stripe.Subscription.retrieve",
            return_value=_subscription_payload(user.user_id),
        ) as retrieve_mock,
    ):
        first = app_client.post(
            "/webhooks/stripe",
            data=b"{}",
            headers={"Stripe-Signature": "t=1,v1=testsig"},
        )
        second = app_client.post(
            "/webhooks/stripe",
            data=b"{}",
            headers={"Stripe-Signature": "t=1,v1=testsig"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert retrieve_mock.call_count == 1


def test_invalid_price_id_returns_422(app_client, pro_price_id):
    _seed_user(app_client)
    response = app_client.post(
        "/billing/checkout", json={"price_id": "price_not_in_allowlist"}
    )
    assert response.status_code == 422
    assert response.json["error"] == "invalid_price_id"


def test_checkout_without_auth_returns_401(app_client):
    response = app_client.post("/billing/checkout", json={"price_id": PRO_PRICE_ID})
    assert response.status_code == 401
    assert response.json["error"] == "unauthorized"
