"""Free-tier lifetime run cap — 24 runs total, alongside the 2/month cap (#91).

Two caps apply to the free tier and they are deliberately distinguishable:

- monthly cap (2)  -> 402 ``plan_limit_exceeded``  — recoverable, resets on the 1st
- lifetime cap (24) -> 402 ``free_tier_exhausted`` — terminal, upgrade is the only path

The lifetime counter lives in the same usage table under the sentinel period key
``TOTAL``, so no infrastructure change is needed and the atomic ADD that makes the
monthly counter concurrency-safe is reused as-is.
"""

import boto3
import pytest
from moto import mock_aws

from scripts.backfill_total_usage import backfill_total_counts

from sanji.service.app import create_app
from sanji.service.billing import DEFAULT_SUBSCRIPTIONS_TABLE
from sanji.service.plans import get_plan
from sanji.service.usage import DEFAULT_USAGE_TABLE, TOTAL_PERIOD_KEY, UsageStore
from sanji.service.users import GOOGLE_SUB_INDEX, UserStore

FREE_LIFETIME_LIMIT = 24
FREE_MONTHLY_LIMIT = 2
JOB_PAYLOAD = {"input_url": "https://youtu.be/test", "params": {}}


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
    # GET /me/usage reads the subscription record alongside the counters
    dynamodb.create_table(
        TableName=DEFAULT_SUBSCRIPTIONS_TABLE,
        AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
        BillingMode="PAY_PER_REQUEST",
    )


@pytest.fixture
def authed_client():
    with mock_aws():
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
        _create_tables(dynamodb)
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(QueueName="sanji-jobs-lifetime-test")["QueueUrl"]
        app = create_app(config_overrides={"SQS_QUEUE_URL": queue_url})
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret"
        user = UserStore().create(
            google_sub="g-lifetime",
            email="lifetime@example.com",
            display_name="Lifetime User",
        )
        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["user_id"] = user.user_id
            yield client, user


def _set_plan(user_id: str, plan_code: str) -> None:
    boto3.resource("dynamodb", region_name="us-west-2").Table(
        "sanji-users"
    ).update_item(
        Key={"user_id": user_id},
        UpdateExpression="SET current_plan_code = :p",
        ExpressionAttributeValues={":p": plan_code},
    )


def _fill_total(user_id: str, count: int) -> None:
    store = UsageStore()
    for _ in range(count):
        store.increment_total_count(user_id)


def _fill_month(user_id: str, count: int) -> None:
    store = UsageStore()
    for _ in range(count):
        store.increment_current_count(user_id)


# ---------------------------------------------------------------------------
# Plan configuration
# ---------------------------------------------------------------------------


def test_free_plan_declares_lifetime_cap() -> None:
    free = get_plan("free")
    assert free is not None
    assert free.total_stream_limit == FREE_LIFETIME_LIMIT
    # the per-month allowance is unchanged by this feature
    assert free.monthly_stream_limit == FREE_MONTHLY_LIMIT


@pytest.mark.parametrize("plan_code", ["pro", "business"])
def test_paid_plans_declare_no_lifetime_cap(plan_code: str) -> None:
    plan = get_plan(plan_code)
    assert plan is not None
    assert plan.total_stream_limit is None


# ---------------------------------------------------------------------------
# Usage store — lifetime counter
# ---------------------------------------------------------------------------


def test_total_counter_increments_and_reads(authed_client) -> None:
    _client, user = authed_client
    store = UsageStore()
    assert store.get_total_count(user.user_id) == 0
    assert store.increment_total_count(user.user_id) == 1
    assert store.increment_total_count(user.user_id) == 2
    assert store.get_total_count(user.user_id) == 2


def test_total_counter_is_independent_of_monthly_counter(authed_client) -> None:
    """The sentinel row must not be swept up by month rollover, or vice versa."""
    _client, user = authed_client
    store = UsageStore()
    store.increment_current_count(user.user_id)
    assert store.get_total_count(user.user_id) == 0

    store.increment_total_count(user.user_id)
    assert store.get_current_count(user.user_id) == 1
    assert store.get_total_count(user.user_id) == 1


def test_total_period_key_cannot_collide_with_a_real_period() -> None:
    """Real keys are always YYYY-MM, so the sentinel is unreachable by rollover."""
    assert TOTAL_PERIOD_KEY == "TOTAL"
    assert not TOTAL_PERIOD_KEY[:4].isdigit()


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


def test_lifetime_exhaustion_blocks_with_distinct_error(authed_client) -> None:
    """AC: lifetime count >= 24 -> 402 free_tier_exhausted, even with month quota free."""
    client, user = authed_client
    _fill_total(user.user_id, FREE_LIFETIME_LIMIT)

    response = client.post("/jobs", json=JOB_PAYLOAD, content_type="application/json")

    assert response.status_code == 402
    assert response.json["error"] == "free_tier_exhausted"
    assert response.json["limit"] == FREE_LIFETIME_LIMIT
    assert response.json["current_count"] == FREE_LIFETIME_LIMIT


def test_lifetime_cap_takes_precedence_over_monthly_cap(authed_client) -> None:
    """Both caps hit -> report the terminal one; telling the user to wait would lie."""
    client, user = authed_client
    _fill_total(user.user_id, FREE_LIFETIME_LIMIT)
    _fill_month(user.user_id, FREE_MONTHLY_LIMIT)

    response = client.post("/jobs", json=JOB_PAYLOAD, content_type="application/json")

    assert response.status_code == 402
    assert response.json["error"] == "free_tier_exhausted"


def test_monthly_cap_still_blocks_while_lifetime_quota_remains(authed_client) -> None:
    """AC: monthly cap unchanged — still 402 plan_limit_exceeded below the ceiling."""
    client, user = authed_client
    _fill_month(user.user_id, FREE_MONTHLY_LIMIT)
    _fill_total(user.user_id, FREE_MONTHLY_LIMIT)

    response = client.post("/jobs", json=JOB_PAYLOAD, content_type="application/json")

    assert response.status_code == 402
    assert response.json["error"] == "plan_limit_exceeded"
    assert response.json["limit"] == FREE_MONTHLY_LIMIT


def test_free_user_under_both_caps_succeeds(authed_client) -> None:
    client, user = authed_client
    _fill_total(user.user_id, FREE_LIFETIME_LIMIT - 1)

    response = client.post("/jobs", json=JOB_PAYLOAD, content_type="application/json")

    assert response.status_code == 201


def test_paid_plan_is_exempt_from_the_lifetime_cap(authed_client) -> None:
    """AC: no lifetime cap on paid plans regardless of accumulated runs."""
    client, user = authed_client
    _set_plan(user.user_id, "pro")
    _fill_total(user.user_id, FREE_LIFETIME_LIMIT * 10)

    response = client.post("/jobs", json=JOB_PAYLOAD, content_type="application/json")

    assert response.status_code == 201


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_successful_submission_increments_both_counters(authed_client) -> None:
    client, user = authed_client

    assert client.post("/jobs", json=JOB_PAYLOAD).status_code == 201

    store = UsageStore()
    assert store.get_current_count(user.user_id) == 1
    assert store.get_total_count(user.user_id) == 1


def test_failed_submission_increments_neither_counter(authed_client) -> None:
    """AC: a rejected submission consumes neither cap."""
    client, user = authed_client

    response = client.post("/jobs", json={"params": {}})  # missing input_url
    assert response.status_code in (400, 422)

    store = UsageStore()
    assert store.get_current_count(user.user_id) == 0
    assert store.get_total_count(user.user_id) == 0


def test_lifetime_cap_trips_from_real_submissions_alone(authed_client) -> None:
    """End-to-end: the ceiling must be reachable without test-code increments.

    Regression guard mirroring #32 — drives the counter to the ceiling through
    real submissions, bumping the monthly cap out of the way between rounds.
    """
    client, user = authed_client
    store = UsageStore()

    for _ in range(FREE_LIFETIME_LIMIT // FREE_MONTHLY_LIMIT):
        assert client.post("/jobs", json=JOB_PAYLOAD).status_code == 201
        assert client.post("/jobs", json=JOB_PAYLOAD).status_code == 201
        # simulate month rollover: the monthly row resets, the TOTAL row does not
        boto3.resource("dynamodb", region_name="us-west-2").Table(
            DEFAULT_USAGE_TABLE
        ).delete_item(
            Key={
                "user_id": user.user_id,
                "period_key": __import__("datetime")
                .datetime.now(__import__("datetime").UTC)
                .strftime("%Y-%m"),
            }
        )

    assert store.get_total_count(user.user_id) == FREE_LIFETIME_LIMIT
    response = client.post("/jobs", json=JOB_PAYLOAD)
    assert response.status_code == 402
    assert response.json["error"] == "free_tier_exhausted"


# ---------------------------------------------------------------------------
# GET /me/usage exposure
# ---------------------------------------------------------------------------


def test_me_usage_reports_lifetime_count_and_limit(authed_client) -> None:
    client, user = authed_client
    _fill_total(user.user_id, 3)
    _fill_month(user.user_id, 1)

    body = client.get("/me/usage").json

    assert body["usage"]["stream_count"] == 1
    assert body["usage"]["stream_limit"] == FREE_MONTHLY_LIMIT
    assert body["usage"]["total_count"] == 3
    assert body["usage"]["total_limit"] == FREE_LIFETIME_LIMIT


def test_me_usage_total_limit_is_null_for_paid_plans(authed_client) -> None:
    client, user = authed_client
    _set_plan(user.user_id, "pro")
    _fill_total(user.user_id, 5)

    body = client.get("/me/usage").json

    assert body["usage"]["total_count"] == 5
    assert body["usage"]["total_limit"] is None


# ---------------------------------------------------------------------------
# Backfill — pre-#91 users have monthly rows but no TOTAL row
# ---------------------------------------------------------------------------


def test_backfill_sums_existing_monthly_rows(authed_client) -> None:
    _client, user = authed_client
    store = UsageStore()
    store.increment_monthly_count(user.user_id, "2026-05")
    store.increment_monthly_count(user.user_id, "2026-06")
    store.increment_monthly_count(user.user_id, "2026-06")
    store.increment_monthly_count(user.user_id, "2026-07")

    totals = backfill_total_counts(DEFAULT_USAGE_TABLE, apply=True)

    assert totals[user.user_id] == 4
    assert store.get_total_count(user.user_id) == 4


def test_backfill_dry_run_writes_nothing(authed_client) -> None:
    _client, user = authed_client
    store = UsageStore()
    store.increment_monthly_count(user.user_id, "2026-06")

    totals = backfill_total_counts(DEFAULT_USAGE_TABLE)

    assert totals[user.user_id] == 1
    assert store.get_total_count(user.user_id) == 0


def test_backfill_is_idempotent(authed_client) -> None:
    """SET (not ADD) semantics — a second run must not double the total."""
    _client, user = authed_client
    store = UsageStore()
    store.increment_monthly_count(user.user_id, "2026-06")
    store.increment_monthly_count(user.user_id, "2026-06")

    backfill_total_counts(DEFAULT_USAGE_TABLE, apply=True)
    backfill_total_counts(DEFAULT_USAGE_TABLE, apply=True)

    assert store.get_total_count(user.user_id) == 2


def test_backfill_excludes_the_sentinel_row_from_the_sum(authed_client) -> None:
    """The TOTAL row must not be counted as if it were a month."""
    _client, user = authed_client
    store = UsageStore()
    store.increment_monthly_count(user.user_id, "2026-06")
    store.increment_total_count(user.user_id)

    totals = backfill_total_counts(DEFAULT_USAGE_TABLE, apply=True)

    assert totals[user.user_id] == 1
    assert store.get_total_count(user.user_id) == 1
