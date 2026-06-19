"""Tests for the sanji service tier (issues #7, #13).

The app is built USING sandjig (extend pattern): sandjig provides /jobs and
/healthcheck; sanji adds /health, /plans, /me and owns the error handlers.
AWS (DynamoDB, SQS) is mocked with moto; sandjig creates its tables at app
construction inside the mock.
"""

import boto3
import pytest
from moto import mock_aws

from sandjig.jobsapi.dyanmodb.models import ProcessingJobModel

from sanji.service.app import create_app
from sanji.service.plans import DEFAULT_PLAN_CODE, PLANS, get_plan
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
    """Tiers decided 2026-06-10 (issue #5)."""
    plans = {p["code"]: p for p in client.get("/plans").json["plans"]}
    assert plans["free"]["price_cents"] == 0
    assert plans["free"]["monthly_stream_limit"] == 2
    assert plans["free"]["max_video_duration_seconds"] == 2 * 60 * 60
    assert plans["pro"]["price_cents"] == 1900
    assert plans["pro"]["monthly_stream_limit"] == 10
    assert plans["pro"]["max_video_duration_seconds"] == 4 * 60 * 60
    assert plans["business"]["price_cents"] == 4900
    assert plans["business"]["monthly_stream_limit"] is None
    assert plans["business"]["max_video_duration_seconds"] == 8 * 60 * 60


def test_me_returns_401_until_oauth_lands(client):
    response = client.get("/me")
    assert response.status_code == 401
    assert response.json["error"] == "unauthorized"


def test_unknown_route_returns_404_json(client):
    """sanji's JSON 404 must override sandjig's plain-text handler."""
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"


def test_job_submit_and_poll(client):
    """POST /jobs records the job (DynamoDB) and queues it (SQS); GET returns it."""
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


def test_job_submit_rejects_missing_input_url(client):
    response = client.post("/jobs", json={"params": {}})
    assert response.status_code in (400, 422)


def test_job_result_returns_presigned_urls(client, monkeypatch):
    """GET /jobs/<id>/result turns stored S3 keys into presigned download URLs (#8)."""
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


def test_job_result_returns_404_for_unknown_job(client):
    response = client.get("/jobs/does-not-exist/result")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"


def test_get_plan_lookup():
    assert get_plan("pro") is not None
    assert get_plan("nonexistent") is None
    assert get_plan(DEFAULT_PLAN_CODE) == PLANS[0]


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
