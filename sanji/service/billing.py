"""Stripe billing integration (issue #4): checkout sessions, webhooks, subscription state.

Subscription and processed-webhook-event records live in DynamoDB (D16: no RDS).
Webhook handlers re-fetch the subscription from Stripe so local state always
reflects Stripe's source of truth, regardless of event delivery order.
"""

import os
from datetime import UTC, datetime
from typing import Any

import boto3
import stripe
import structlog
from flask import request
from pydantic import BaseModel

from sanji.service.auth import CurrentUser
from sanji.service.plans import PLANS, resolve_stripe_price_id
from sanji.service.users import UserStore

logger = structlog.get_logger().bind(logger=__name__)

SUBSCRIPTIONS_TABLE_ENV = "SANJI_SUBSCRIPTIONS_TABLE"
DEFAULT_SUBSCRIPTIONS_TABLE = "sanji-subscriptions"
WEBHOOK_EVENTS_TABLE_ENV = "SANJI_WEBHOOK_EVENTS_TABLE"
DEFAULT_WEBHOOK_EVENTS_TABLE = "sanji-webhook-events"
BASE_URL_ENV = "SANJI_BASE_URL"
DEFAULT_BASE_URL = "http://localhost:5000"

# Stripe reports "canceled"; the local record uses "cancelled".
_STRIPE_STATUS_MAP = {"canceled": "cancelled"}


class InvalidPriceIdError(ValueError):
    """price_id is not in the PLANS allowlist."""


class SubscriptionRecord(BaseModel):
    user_id: str
    stripe_subscription_id: str
    stripe_customer_id: str
    plan_code: str
    status: str
    # ISO datetime; None when Stripe omits the period end (informational field)
    current_period_end: str | None = None
    updated_at: str


class ProcessedEventRecord(BaseModel):
    stripe_event_id: str
    processed_at: str


STRIPE_CUSTOMER_INDEX = "stripe_customer_id-index"


class SubscriptionStore:
    """Owns SubscriptionRecord persistence in DynamoDB (keyed by user_id)."""

    def __init__(self, table_name: str | None = None) -> None:
        self._table_name = table_name or os.getenv(
            SUBSCRIPTIONS_TABLE_ENV, DEFAULT_SUBSCRIPTIONS_TABLE
        )
        self._table = boto3.resource("dynamodb").Table(self._table_name)

    def put(self, record: SubscriptionRecord) -> None:
        self._table.put_item(Item=record.model_dump())

    def get(self, user_id: str) -> SubscriptionRecord | None:
        response = self._table.get_item(Key={"user_id": user_id})
        item: dict[str, Any] | None = response.get("Item")
        return SubscriptionRecord.model_validate(item) if item else None

    def get_by_customer_id(self, stripe_customer_id: str) -> SubscriptionRecord | None:
        """Keyed GSI query — this runs on every subscription/invoice webhook.

        The previous Scan+FilterExpression cost grew with total table size and
        silently missed matches beyond the 1MB scan page (#40).
        """
        response = self._table.query(
            IndexName=STRIPE_CUSTOMER_INDEX,
            KeyConditionExpression="stripe_customer_id = :cid",
            ExpressionAttributeValues={":cid": stripe_customer_id},
            Limit=1,
        )
        items = response.get("Items", [])
        return SubscriptionRecord.model_validate(items[0]) if items else None


class ProcessedEventStore:
    """Webhook idempotency ledger keyed by stripe_event_id."""

    def __init__(self, table_name: str | None = None) -> None:
        self._table_name = table_name or os.getenv(
            WEBHOOK_EVENTS_TABLE_ENV, DEFAULT_WEBHOOK_EVENTS_TABLE
        )
        self._table = boto3.resource("dynamodb").Table(self._table_name)

    def is_processed(self, event_id: str) -> bool:
        response = self._table.get_item(Key={"stripe_event_id": event_id})
        return response.get("Item") is not None

    def mark_processed(self, event_id: str) -> None:
        record = ProcessedEventRecord(
            stripe_event_id=event_id,
            processed_at=datetime.now(UTC).isoformat(),
        )
        self._table.put_item(Item=record.model_dump())


def _plan_code_for_price_id(price_id: str) -> str | None:
    return next(
        (plan.code for plan in PLANS if resolve_stripe_price_id(plan) == price_id),
        None,
    )


class BillingService:
    """Orchestrates Stripe checkout and webhook-driven subscription sync."""

    def __init__(
        self,
        user_store: UserStore | None = None,
        subscription_store: SubscriptionStore | None = None,
        processed_event_store: ProcessedEventStore | None = None,
    ) -> None:
        self._users = user_store or UserStore()
        self._subscriptions = subscription_store or SubscriptionStore()
        self._processed_events = processed_event_store or ProcessedEventStore()

    def create_checkout_session(self, user: CurrentUser, price_id: str) -> str:
        """Create a fresh Checkout Session (no idempotency key: a static key made
        Stripe return the same — possibly expired or completed — session for 24h
        when a user retried after abandoning checkout; creation is safe to repeat).
        """
        if _plan_code_for_price_id(price_id) is None:
            raise InvalidPriceIdError(f"price_id not in PLANS allowlist: {price_id}")

        stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
        customer_id = self._get_or_create_customer(user)

        base_url = os.getenv(BASE_URL_ENV, DEFAULT_BASE_URL)
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            client_reference_id=user.user_id,
            subscription_data={"metadata": {"user_id": user.user_id}},
            success_url=f"{base_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{base_url}/billing/cancel",
        )
        logger.info("checkout_session_created", user_id=user.user_id, price_id=price_id)
        return session.url

    def _get_or_create_customer(self, user: CurrentUser) -> str:
        full_user = self._users.get(user.user_id)
        if full_user and full_user.stripe_customer_id:
            return full_user.stripe_customer_id

        customer = stripe.Customer.create(
            email=user.email, metadata={"user_id": user.user_id}
        )
        self._users.update_stripe_customer(user.user_id, stripe_customer_id=customer.id)
        logger.info(
            "stripe_customer_created",
            user_id=user.user_id,
            stripe_customer_id=customer.id,
        )
        return customer.id

    def handle_webhook(self, payload: bytes, sig_header: str) -> None:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET", "")
        )
        event_id = event["id"]
        if self._processed_events.is_processed(event_id):
            logger.info("webhook_event_duplicate", event_id=event_id)
            return

        handler = self._event_handlers().get(event["type"])
        if handler is None:
            logger.info(
                "webhook_event_ignored", event_id=event_id, event_type=event["type"]
            )
        else:
            handler(event)
        self._processed_events.mark_processed(event_id)

    def _event_handlers(self) -> dict[str, Any]:
        return {
            "customer.subscription.created": self._handle_subscription_updated,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "invoice.payment_succeeded": self._handle_invoice_payment_succeeded,
            "invoice.payment_failed": self._handle_invoice_payment_failed,
        }

    def _handle_subscription_updated(self, event: dict[str, Any]) -> None:
        self._sync_subscription(event["data"]["object"]["id"])

    def _handle_subscription_deleted(self, event: dict[str, Any]) -> None:
        self._sync_subscription(event["data"]["object"]["id"])

    def _handle_invoice_payment_succeeded(self, event: dict[str, Any]) -> None:
        self._sync_invoice_subscription(event)

    def _handle_invoice_payment_failed(self, event: dict[str, Any]) -> None:
        self._sync_invoice_subscription(event)

    def _sync_invoice_subscription(self, event: dict[str, Any]) -> None:
        subscription_id = event["data"]["object"].get("subscription")
        if not subscription_id:
            logger.info("invoice_without_subscription", event_id=event["id"])
            return
        self._sync_subscription(subscription_id)

    def _sync_subscription(self, subscription_id: str) -> None:
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
        subscription = stripe.Subscription.retrieve(subscription_id)
        customer_id = subscription["customer"]

        user_id = self._resolve_user_id(subscription, customer_id)
        if user_id is None:
            logger.warning(
                "subscription_user_unresolved",
                subscription_id=subscription_id,
                stripe_customer_id=customer_id,
            )
            return

        items = (subscription.get("items") or {}).get("data") or []
        price_id = items[0]["price"]["id"] if items else ""
        status = subscription["status"]
        # Stripe API 2025-03-31.basil moved current_period_end from the
        # subscription object to its items; fall back to the legacy top-level
        # field for older-API payloads (#36).
        period_end_ts = (
            items[0].get("current_period_end") if items else None
        ) or subscription.get("current_period_end")
        if period_end_ts is None:
            logger.warning(
                "subscription_period_end_missing", subscription_id=subscription_id
            )
        record = SubscriptionRecord(
            user_id=user_id,
            stripe_subscription_id=subscription["id"],
            stripe_customer_id=customer_id,
            plan_code=_plan_code_for_price_id(price_id) or "unknown",
            status=_STRIPE_STATUS_MAP.get(status, status),
            current_period_end=datetime.fromtimestamp(period_end_ts, tz=UTC).isoformat()
            if period_end_ts
            else None,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._subscriptions.put(record)
        logger.info(
            "subscription_synced",
            user_id=user_id,
            stripe_subscription_id=record.stripe_subscription_id,
            status=record.status,
            plan_code=record.plan_code,
        )

    def _resolve_user_id(self, subscription: Any, customer_id: str) -> str | None:
        existing = self._subscriptions.get_by_customer_id(customer_id)
        if existing:
            return existing.user_id
        metadata = subscription.get("metadata") or {}
        return metadata.get("user_id")


# ---------------------------------------------------------------------------
# Route functions (registered in app.py)
# ---------------------------------------------------------------------------


def handle_checkout(
    current_user: CurrentUser, billing_service: BillingService
) -> tuple[dict, int]:
    body = request.get_json(silent=True) or {}
    price_id = body.get("price_id")
    if not price_id:
        return {"error": "invalid_price_id", "message": "price_id is required."}, 422

    try:
        url = billing_service.create_checkout_session(current_user, price_id)
    except InvalidPriceIdError:
        logger.info(
            "checkout_rejected_invalid_price_id",
            user_id=current_user.user_id,
            price_id=price_id,
        )
        return {"error": "invalid_price_id"}, 422
    return {"url": url}, 200


def handle_stripe_webhook(billing_service: BillingService) -> tuple[dict, int]:
    # Raw body required — signature is computed over the exact bytes Stripe sent.
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")
    try:
        billing_service.handle_webhook(payload, sig_header)
    except (ValueError, stripe.SignatureVerificationError) as exc:
        logger.warning("webhook_signature_invalid", error=str(exc))
        return {"error": "invalid_signature"}, 400
    return {"status": "ok"}, 200
