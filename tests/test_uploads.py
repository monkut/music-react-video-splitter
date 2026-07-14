"""Direct-upload API + source_s3_key job flow tests (issue #65).

POST /uploads issues a presigned PUT (small files) or a multipart part-URL set
(multi-GB) scoped under ``uploads/<user_id>/<upload_id>/`` — the bytes never
pass through the API. POST /uploads/complete finishes a multipart upload after
verifying ownership and enforcing the size cap. POST /jobs accepts
``source_s3_key`` as an alternative to ``input_url``; ownership is enforced at
the API and re-validated on the model (worker defense-in-depth).
"""

import json
import math

import boto3
import pytest
from moto import mock_aws
from pydantic import ValidationError

from sanji.service.app import create_app
from sanji.service.jobs import SanjiJobRequest, is_user_upload_key
from sanji.service.uploads import CONTENT_TYPE_EXTENSIONS
from sanji.service.usage import DEFAULT_USAGE_TABLE
from sanji.service.users import GOOGLE_SUB_INDEX, UserStore
from sanji.settings import (
    UPLOAD_MULTIPART_THRESHOLD_BYTES,
    UPLOAD_PART_BYTES,
    UPLOAD_PRESIGN_EXPIRY_SECONDS,
    UPLOADS_BUCKET_ENV,
    UPLOADS_ROOT,
)
from sanji.worker import parse_job_message

UPLOADS_BUCKET = "sanji-uploads-test"


@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")


@pytest.fixture
def tables():
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
def authed_client(tables, monkeypatch):
    """Authenticated test client with the uploads bucket + env in place."""
    monkeypatch.setenv(UPLOADS_BUCKET_ENV, UPLOADS_BUCKET)
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(
            Bucket=UPLOADS_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(QueueName="sanji-jobs-uploads-test")["QueueUrl"]
        app = create_app(config_overrides={"SQS_QUEUE_URL": queue_url})
        app.config["TESTING"] = True

        user = UserStore().create(
            google_sub="g-uploader",
            email="uploader@example.com",
            display_name="Uploader",
        )

        with app.test_client() as c:
            with c.session_transaction() as sess:
                sess["user_id"] = user.user_id
            yield c, user, s3


SMALL_UPLOAD = {
    "filename": "stream.mp4",
    "content_type": "video/mp4",
    "content_length": 10 * 1024 * 1024,
}


# ---------------------------------------------------------------------------
# POST /uploads
# ---------------------------------------------------------------------------


def test_create_upload_requires_auth(tables):
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(QueueName="sanji-jobs-anon-test")["QueueUrl"]
        app = create_app(config_overrides={"SQS_QUEUE_URL": queue_url})
        app.config["TESTING"] = True
        response = app.test_client().post("/uploads", json=SMALL_UPLOAD)
    assert response.status_code == 401


def test_create_upload_returns_user_scoped_presigned_put(authed_client):
    client, user, _s3 = authed_client
    response = client.post("/uploads", json=SMALL_UPLOAD)
    assert response.status_code == 201, response.data
    body = response.json

    assert body["method"] == "put"
    assert body["key"].startswith(f"{UPLOADS_ROOT}/{user.user_id}/")
    assert body["key"].endswith("/source.mp4")
    assert body["upload_id"] in body["key"]
    assert body["expires_in"] <= UPLOAD_PRESIGN_EXPIRY_SECONDS
    assert body["key"] in body["url"] or body["upload_id"] in body["url"]
    # the client must PUT with exactly the declared type/length (signed headers)
    assert body["headers"]["Content-Type"] == "video/mp4"
    assert body["headers"]["Content-Length"] == SMALL_UPLOAD["content_length"]


def test_create_upload_rejects_disallowed_content_type(authed_client):
    client, _user, _s3 = authed_client
    response = client.post(
        "/uploads",
        json={**SMALL_UPLOAD, "content_type": "application/x-sh"},
    )
    assert response.status_code == 415


def test_create_upload_rejects_over_max_content_length(authed_client, monkeypatch):
    monkeypatch.setattr("sanji.service.uploads.UPLOAD_MAX_BYTES", 1024)
    client, _user, _s3 = authed_client
    response = client.post("/uploads", json={**SMALL_UPLOAD, "content_length": 2048})
    assert response.status_code == 413


def test_create_upload_rejects_invalid_payload(authed_client):
    client, _user, _s3 = authed_client
    response = client.post("/uploads", json={"content_type": "video/mp4"})
    assert response.status_code == 422


def test_create_upload_multipart_for_large_files(authed_client):
    client, user, s3 = authed_client
    content_length = UPLOAD_MULTIPART_THRESHOLD_BYTES + 2 * UPLOAD_PART_BYTES
    response = client.post(
        "/uploads", json={**SMALL_UPLOAD, "content_length": content_length}
    )
    assert response.status_code == 201, response.data
    body = response.json

    assert body["method"] == "multipart"
    assert body["key"].startswith(f"{UPLOADS_ROOT}/{user.user_id}/")
    assert body["part_size"] == UPLOAD_PART_BYTES
    assert body["s3_upload_id"]
    expected_parts = math.ceil(content_length / UPLOAD_PART_BYTES)
    assert [p["part_number"] for p in body["part_urls"]] == list(
        range(1, expected_parts + 1)
    )
    assert all(body["key"] in p["url"] for p in body["part_urls"])
    # the multipart upload was actually opened against S3
    pending = s3.list_multipart_uploads(Bucket=UPLOADS_BUCKET).get("Uploads", [])
    assert body["s3_upload_id"] in [u["UploadId"] for u in pending]


# ---------------------------------------------------------------------------
# POST /uploads/complete
# ---------------------------------------------------------------------------


def _start_multipart(client, content_length):
    response = client.post(
        "/uploads", json={**SMALL_UPLOAD, "content_length": content_length}
    )
    assert response.status_code == 201, response.data
    return response.json


def test_complete_multipart_upload(authed_client):
    client, _user, s3 = authed_client
    started = _start_multipart(client, UPLOAD_MULTIPART_THRESHOLD_BYTES + 1)

    part = s3.upload_part(
        Bucket=UPLOADS_BUCKET,
        Key=started["key"],
        PartNumber=1,
        UploadId=started["s3_upload_id"],
        Body=b"video-bytes",
    )
    response = client.post(
        "/uploads/complete",
        json={
            "key": started["key"],
            "s3_upload_id": started["s3_upload_id"],
            "parts": [{"part_number": 1, "etag": part["ETag"]}],
        },
    )
    assert response.status_code == 200, response.data
    assert response.json["key"] == started["key"]
    stored = s3.head_object(Bucket=UPLOADS_BUCKET, Key=started["key"])
    assert stored["ContentLength"] == len(b"video-bytes")


def test_complete_rejects_foreign_key(authed_client):
    client, _user, _s3 = authed_client
    response = client.post(
        "/uploads/complete",
        json={
            "key": f"{UPLOADS_ROOT}/other-user/abc/source.mp4",
            "s3_upload_id": "irrelevant",
            "parts": [{"part_number": 1, "etag": "x"}],
        },
    )
    assert response.status_code == 403


def test_complete_rejects_invalid_parts(authed_client):
    """A complete with parts S3 doesn't recognise returns 400, not 500.

    (An unknown s3_upload_id takes the same ClientError path on real S3, but
    moto raises a raw KeyError for that case, so invalid parts stand in.)
    """
    client, _user, _s3 = authed_client
    started = _start_multipart(client, UPLOAD_MULTIPART_THRESHOLD_BYTES + 1)
    response = client.post(
        "/uploads/complete",
        json={
            "key": started["key"],
            "s3_upload_id": started["s3_upload_id"],
            "parts": [{"part_number": 1, "etag": "bogus-etag"}],
        },
    )
    assert response.status_code == 400


def test_complete_deletes_object_exceeding_max_bytes(authed_client, monkeypatch):
    monkeypatch.setattr("sanji.service.uploads.UPLOAD_MAX_BYTES", 1024)
    monkeypatch.setattr("sanji.service.uploads.UPLOAD_MULTIPART_THRESHOLD_BYTES", 16)
    client, _user, s3 = authed_client
    started = _start_multipart(client, 512)  # declared under the cap

    part = s3.upload_part(
        Bucket=UPLOADS_BUCKET,
        Key=started["key"],
        PartNumber=1,
        UploadId=started["s3_upload_id"],
        Body=b"x" * 2048,  # actual bytes exceed the cap
    )
    response = client.post(
        "/uploads/complete",
        json={
            "key": started["key"],
            "s3_upload_id": started["s3_upload_id"],
            "parts": [{"part_number": 1, "etag": part["ETag"]}],
        },
    )
    assert response.status_code == 413
    listed = s3.list_objects_v2(Bucket=UPLOADS_BUCKET, Prefix=started["key"])
    assert listed.get("KeyCount", 0) == 0  # oversize object was removed


# ---------------------------------------------------------------------------
# POST /jobs with source_s3_key
# ---------------------------------------------------------------------------


def test_job_submit_with_owned_source_s3_key(authed_client):
    client, user, _s3 = authed_client
    key = f"{UPLOADS_ROOT}/{user.user_id}/up1/source.mp4"
    response = client.post("/jobs", json={"source_s3_key": key})
    assert response.status_code == 201, response.data
    assert response.json["request_payload"]["source_s3_key"] == key


def test_job_submit_rejects_foreign_source_s3_key(authed_client):
    client, _user, _s3 = authed_client
    response = client.post(
        "/jobs",
        json={"source_s3_key": f"{UPLOADS_ROOT}/someone-else/up1/source.mp4"},
    )
    assert response.status_code == 403


def test_job_submit_rejects_both_input_url_and_source_key(authed_client):
    client, user, _s3 = authed_client
    response = client.post(
        "/jobs",
        json={
            "input_url": "https://youtu.be/x",
            "source_s3_key": f"{UPLOADS_ROOT}/{user.user_id}/up1/source.mp4",
        },
    )
    assert response.status_code == 422


def test_job_submit_rejects_neither_input(authed_client):
    client, _user, _s3 = authed_client
    response = client.post("/jobs", json={"params": {}})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Model-level validation (worker defense-in-depth)
# ---------------------------------------------------------------------------


def test_request_accepts_source_s3_key_alone():
    request = SanjiJobRequest(source_s3_key="uploads/u1/up1/source.mp4")
    assert request.input_url is None


@pytest.mark.parametrize(
    "payload",
    [
        {"input_url": "https://youtu.be/x", "source_s3_key": "uploads/u/a/s.mp4"},
        {},
        {"params": {"threshold": 0.4}},
    ],
)
def test_request_requires_exactly_one_source(payload):
    with pytest.raises(ValidationError):
        SanjiJobRequest(**payload)


@pytest.mark.parametrize(
    "key",
    [
        "results/u1/up1/source.mp4",  # wrong root
        "uploads/u1/source.mp4",  # missing upload_id segment
        "uploads/u1/../u2/source.mp4",  # traversal
        "uploads//up1/source.mp4",  # empty user segment
        "/uploads/u1/up1/source.mp4",  # absolute
        "uploads/u1/up1/a/b.mp4",  # extra depth
    ],
)
def test_request_rejects_malformed_source_keys(key):
    with pytest.raises(ValidationError):
        SanjiJobRequest(source_s3_key=key)


def test_request_rejects_source_key_owned_by_other_user():
    with pytest.raises(ValidationError):
        SanjiJobRequest(source_s3_key="uploads/alice/up1/source.mp4", user_id="bob")


def test_worker_rejects_injected_foreign_source_key_message():
    """A message injected past the API must not let one user read another's upload."""
    raw = json.dumps(
        {
            "job_id": "j-1",
            "request_payload": {
                "source_s3_key": "uploads/alice/up1/source.mp4",
                "user_id": "bob",
            },
        }
    )
    with pytest.raises(ValidationError):
        parse_job_message(raw)


def test_is_user_upload_key():
    assert is_user_upload_key("uploads/u1/up1/source.mp4", "u1")
    assert not is_user_upload_key("uploads/u1/up1/source.mp4", "u2")
    assert not is_user_upload_key("uploads/u1/../u2/source.mp4", "u1")
    assert not is_user_upload_key("results/u1/up1/source.mp4", "u1")


def test_content_type_allowlist_covers_common_video_types():
    for content_type in ("video/mp4", "video/webm", "video/x-matroska"):
        assert content_type in CONTENT_TYPE_EXTENSIONS
