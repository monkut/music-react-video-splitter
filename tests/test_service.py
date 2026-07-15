"""Tests for the sanji service tier (issues #7, #13).

The app is built USING sandjig (extend pattern): sandjig provides /jobs and
/healthcheck; sanji adds /health, /plans, /me and owns the error handlers.
AWS (DynamoDB, SQS) is mocked with moto; sandjig creates its tables at app
construction inside the mock.
"""

import boto3
import pytest
from moto import mock_aws

from sandjig.jobsapi.dynamodb.models import ProcessingJobModel

from sanji.service.app import create_app
from sanji.service.plans import DEFAULT_PLAN_CODE, PLANS, get_plan
from sanji.service.usage import DEFAULT_USAGE_TABLE, UsageStore
from sanji.service.users import GOOGLE_SUB_INDEX, UserStore

RESULTS_BUCKET = "sanji-results-test"


@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")


@pytest.fixture
def client():
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(QueueName="sanji-jobs-test")["QueueUrl"]
        app = create_app(config_overrides={"SQS_QUEUE_URL": queue_url})
        app.config["TESTING"] = True
        yield app.test_client()


@pytest.fixture
def users_table():
    with mock_aws():
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
        yield


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert "version" in response.json


def test_sandjig_healthcheck_registered(client):
    response = client.get("/healthcheck")
    assert response.status_code == 200


def test_plans_lists_three_tiers(client):
    response = client.get("/plans")
    assert response.status_code == 200
    plans = response.json["plans"]
    assert [p["code"] for p in plans] == ["free", "pro", "business"]


def test_plans_match_decided_tiers(client):
    """Tiers decided 2026-06-10; updated 2026-07-01 (issue #5)."""
    plans = {p["code"]: p for p in client.get("/plans").json["plans"]}
    two_hours = 2 * 60 * 60
    assert plans["free"]["price_cents"] == 0
    assert plans["free"]["monthly_stream_limit"] == 2
    assert plans["free"]["max_video_duration_seconds"] == two_hours
    assert plans["pro"]["price_cents"] == 1900
    assert plans["pro"]["monthly_stream_limit"] == 10
    assert plans["pro"]["max_video_duration_seconds"] == two_hours
    assert plans["business"]["price_cents"] == 3900
    assert plans["business"]["monthly_stream_limit"] == 30
    assert plans["business"]["max_video_duration_seconds"] == two_hours


def test_me_returns_401_until_oauth_lands(client):
    response = client.get("/me")
    assert response.status_code == 401
    assert response.json["error"] == "unauthorized"


def test_unknown_route_returns_404_json(client):
    """sanji's JSON 404 must override sandjig's plain-text handler."""
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"


def test_job_submit_and_poll(authed_client):
    """POST /jobs records the job (DynamoDB) and queues it (SQS); GET returns it."""
    client, user = authed_client
    payload = {
        "input_url": "https://www.youtube.com/watch?v=TESTVIDEO01",
        "params": {"threshold": 0.5},
    }
    response = client.post("/jobs", json=payload)
    assert response.status_code == 201, response.data
    body = response.json
    job_id = body["job_id"]
    assert body["status"] == "queued"
    assert body["request_payload"]["input_url"] == payload["input_url"]

    poll = client.get(f"/jobs/{job_id}")
    assert poll.status_code == 200
    assert poll.json["job_id"] == job_id


def test_job_status_readable_after_error(authed_client):
    """GET /jobs/<id> must stay readable once a job is terminal-error (#74).

    sanji wrote ``errors`` as a dict while sandjig's JobResponse declares
    ``list[str] | None`` — the response validation 500'd exactly when the
    client needed the error detail.
    """
    from sanji.service.jobs import STATUS_ERROR
    from sanji.service.results import _record_terminal_status

    client, _user = authed_client
    response = client.post(
        "/jobs",
        json={"input_url": "https://www.youtube.com/watch?v=TESTVIDEO01"},
    )
    assert response.status_code == 201, response.data
    job_id = response.json["job_id"]

    _record_terminal_status(job_id, STATUS_ERROR, error="ffmpeg exited 1")

    poll = client.get(f"/jobs/{job_id}")
    assert poll.status_code == 200, poll.data
    assert poll.json["status"] == STATUS_ERROR
    assert poll.json["errors"] == ["ffmpeg exited 1"]


def test_job_submit_rejects_missing_input_url(authed_client):
    client, _user = authed_client
    response = client.post("/jobs", json={"params": {}})
    assert response.status_code in (400, 422)


def test_job_result_returns_presigned_urls(authed_client, monkeypatch):
    """GET /jobs/<id>/result turns stored S3 keys into presigned download URLs (#8)."""
    client, user = authed_client
    monkeypatch.setenv("SANJI_RESULTS_BUCKET", RESULTS_BUCKET)
    boto3.client("s3", region_name="us-west-2").create_bucket(
        Bucket=RESULTS_BUCKET,
        CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
    )
    job_id = client.post("/jobs", json={"input_url": "https://youtu.be/x"}).json[
        "job_id"
    ]
    job = ProcessingJobModel.get_processingjobmodel_item(job_id, as_dict=False)
    job.update(
        actions=[
            ProcessingJobModel.response_payload.set(
                {
                    "segment_keys": [f"results/{job_id}/segments/segment_00.mp4"],
                    "result_manifest_key": f"results/{job_id}/manifest.csv",
                }
            )
        ]
    )

    response = client.get(f"/jobs/{job_id}/result")

    assert response.status_code == 200
    assert response.json["job_id"] == job_id
    assert len(response.json["segment_urls"]) == 1
    assert RESULTS_BUCKET in response.json["segment_urls"][0]
    assert response.json["manifest_url"] is not None


def test_job_result_returns_404_for_unknown_job(authed_client):
    client, _user = authed_client
    response = client.get("/jobs/does-not-exist/result")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"


# ---------------------------------------------------------------------------
# Jobs API authentication, attribution, ownership (#30)
# ---------------------------------------------------------------------------


def test_anonymous_job_submit_returns_401(client):
    """Job submission consumes metered compute — anonymous POST /jobs is rejected."""
    response = client.post(
        "/jobs",
        json={"input_url": "https://youtu.be/test"},
        content_type="application/json",
    )
    assert response.status_code == 401
    assert response.json["error"] == "unauthorized"


def test_anonymous_job_result_returns_401(client):
    response = client.get("/jobs/any-id/result")
    assert response.status_code == 401
    assert response.json["error"] == "unauthorized"


def test_client_supplied_user_id_is_overwritten(authed_client):
    """user_id is set server-side from the session; spoofed values are ignored."""
    client, user = authed_client
    response = client.post(
        "/jobs",
        json={"input_url": "https://youtu.be/test", "user_id": "attacker-chosen"},
        content_type="application/json",
    )
    assert response.status_code == 201
    assert response.json["request_payload"]["user_id"] == user.user_id


def test_job_attributed_to_session_user(authed_client):
    client, user = authed_client
    response = client.post(
        "/jobs",
        json={"input_url": "https://youtu.be/test"},
        content_type="application/json",
    )
    assert response.status_code == 201
    assert response.json["request_payload"]["user_id"] == user.user_id


def test_job_result_denied_for_non_owner(authed_client):
    """A non-owner receives 404 — job ids must not be confirmable by probing."""
    client, user = authed_client
    job_id = client.post(
        "/jobs",
        json={"input_url": "https://youtu.be/test"},
        content_type="application/json",
    ).json["job_id"]

    other = UserStore().create(
        google_sub="g-other", email="other@example.com", display_name="Other"
    )
    with client.session_transaction() as sess:
        sess["user_id"] = other.user_id

    response = client.get(f"/jobs/{job_id}/result")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"


# ---------------------------------------------------------------------------
# Usage counting on successful submission (#32)
# ---------------------------------------------------------------------------


def test_successful_submission_increments_usage(authed_client):
    client, user = authed_client
    period_key = (
        __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .strftime("%Y-%m")
    )
    assert UsageStore().get_monthly_count(user.user_id, period_key) == 0

    response = client.post(
        "/jobs",
        json={"input_url": "https://youtu.be/test"},
        content_type="application/json",
    )
    assert response.status_code == 201
    assert UsageStore().get_monthly_count(user.user_id, period_key) == 1


def test_failed_submission_does_not_increment_usage(authed_client):
    client, user = authed_client
    period_key = (
        __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .strftime("%Y-%m")
    )

    response = client.post("/jobs", json={"params": {}})  # missing input_url
    assert response.status_code in (400, 422)
    assert UsageStore().get_monthly_count(user.user_id, period_key) == 0


def test_plan_limit_enforced_end_to_end_without_manual_increments(authed_client):
    """Free plan (limit 2): two real submissions succeed, the third is blocked.

    Regression for #32 — enforcement must trip from real submissions alone;
    previously the counter was only ever incremented by test code.
    """
    client, user = authed_client
    payload = {"input_url": "https://youtu.be/test", "params": {}}

    assert client.post("/jobs", json=payload).status_code == 201
    assert client.post("/jobs", json=payload).status_code == 201

    response = client.post("/jobs", json=payload)
    assert response.status_code == 402
    assert response.json["error"] == "plan_limit_exceeded"


@pytest.fixture
def usage_table():
    with mock_aws():
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
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
        yield


@pytest.fixture
def authed_client(users_table, usage_table):
    """Test client with a logged-in user whose session is pre-populated."""
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(QueueName="sanji-jobs-authed-test")["QueueUrl"]
        app = create_app(config_overrides={"SQS_QUEUE_URL": queue_url})
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret"

        store = UserStore()
        user = store.create(
            google_sub="g-authed",
            email="authed@example.com",
            display_name="Authed User",
        )

        with app.test_client() as c:
            with c.session_transaction() as sess:
                sess["user_id"] = user.user_id
            yield c, user


def test_get_plan_lookup():
    assert get_plan("pro") is not None
    assert get_plan("nonexistent") is None
    assert get_plan(DEFAULT_PLAN_CODE) == PLANS[0]


# ---------------------------------------------------------------------------
# Usage store unit tests
# ---------------------------------------------------------------------------


def test_usage_store_increment_and_get(usage_table):
    store = UsageStore()
    assert store.get_monthly_count("u1", "2026-07") == 0
    store.increment_monthly_count("u1", "2026-07")
    assert store.get_monthly_count("u1", "2026-07") == 1
    store.increment_monthly_count("u1", "2026-07")
    assert store.get_monthly_count("u1", "2026-07") == 2


def test_usage_store_independent_periods(usage_table):
    store = UsageStore()
    store.increment_monthly_count("u1", "2026-06")
    assert store.get_monthly_count("u1", "2026-07") == 0
    assert store.get_monthly_count("u1", "2026-06") == 1


# ---------------------------------------------------------------------------
# Plan enforcement — enforcement point A (monthly count gate, HTTP 402)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plan_code,stream_limit",
    [
        ("free", 2),
        ("pro", 10),
        ("business", 30),
    ],
)
def test_plan_limit_enforcement_blocks_at_cap(authed_client, plan_code, stream_limit):
    """Enforcement A: POST /jobs returns 402 when monthly limit is reached."""
    client, user = authed_client
    period_key = (
        __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .strftime("%Y-%m")
    )

    # Update user to the target plan.
    boto3.resource("dynamodb", region_name="us-west-2").Table(
        "sanji-users"
    ).update_item(
        Key={"user_id": user.user_id},
        UpdateExpression="SET current_plan_code = :p",
        ExpressionAttributeValues={":p": plan_code},
    )

    # Fill the usage counter to the cap.
    usage_store = UsageStore()
    for _ in range(stream_limit):
        usage_store.increment_monthly_count(user.user_id, period_key)

    response = client.post(
        "/jobs",
        json={"input_url": "https://youtu.be/test", "params": {}},
        content_type="application/json",
    )
    assert response.status_code == 402
    assert response.json["error"] == "plan_limit_exceeded"
    assert response.json["limit"] == stream_limit


def test_plan_limit_allows_under_cap(authed_client):
    """Enforcement A: POST /jobs succeeds when under the monthly limit."""
    client, user = authed_client
    response = client.post(
        "/jobs",
        json={"input_url": "https://youtu.be/test", "params": {}},
        content_type="application/json",
    )
    assert response.status_code == 201


def test_user_store_create_and_get(users_table):
    store = UserStore()
    created = store.create(google_sub="g-123", email="a@example.com", display_name="A")
    assert created.current_plan_code == DEFAULT_PLAN_CODE

    fetched = store.get(created.user_id)
    assert fetched is not None
    assert fetched.google_sub == "g-123"

    by_sub = store.get_by_google_sub("g-123")
    assert by_sub is not None
    assert by_sub.user_id == created.user_id

    assert store.get("missing") is None
    assert store.get_by_google_sub("missing") is None
