"""Credentialed cross-origin CORS tests (#81).

sandjig sets ``Access-Control-Allow-Origin: *`` unconditionally; browsers reject a
wildcard origin on credentialed (cookie) requests, so the cookie-authenticated
SPA cannot call the API cross-origin. sanji wraps the WSGI app with
``CorsCredentialsMiddleware`` to echo an allow-listed origin + credentials and to
deny every other origin.
"""

import boto3
import pytest
from moto import mock_aws

from sanji.service.app import create_app
from sanji.settings import CORS_ALLOWED_ORIGINS_ENV

ALLOWED_ORIGIN = "https://kanpaiko.weyuco.com"
DISALLOWED_ORIGIN = "https://evil.example"


@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(CORS_ALLOWED_ORIGINS_ENV, ALLOWED_ORIGIN)
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(QueueName="sanji-jobs-test")["QueueUrl"]
        app = create_app(config_overrides={"SQS_QUEUE_URL": queue_url})
        app.config["TESTING"] = True
        yield app.test_client()


def test_allowlisted_origin_echoed_with_credentials(client) -> None:
    resp = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert resp.status_code == 200
    assert resp.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert resp.headers["Access-Control-Allow-Credentials"] == "true"


def test_allowlisted_origin_is_not_wildcard(client) -> None:
    resp = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert resp.headers["Access-Control-Allow-Origin"] != "*"


def test_vary_origin_is_set_for_caches(client) -> None:
    resp = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert "Origin" in resp.headers.get("Vary", "")


def test_credentialed_methods_are_explicit_not_wildcard(client) -> None:
    # `Access-Control-Allow-Methods: *` is invalid on a credentialed response.
    resp = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert resp.headers.get("Access-Control-Allow-Methods") != "*"


def test_disallowed_origin_denied(client) -> None:
    resp = client.get("/health", headers={"Origin": DISALLOWED_ORIGIN})
    assert "Access-Control-Allow-Origin" not in resp.headers
    assert "Access-Control-Allow-Credentials" not in resp.headers


def test_preflight_options_allowlisted_origin(client) -> None:
    resp = client.options("/jobs", headers={"Origin": ALLOWED_ORIGIN})
    assert resp.status_code == 200
    assert resp.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert resp.headers["Access-Control-Allow-Credentials"] == "true"


def test_no_origin_request_unaffected(client) -> None:
    # A non-CORS (same-origin / non-browser) request has no Origin header and
    # must not be broken by the middleware.
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "Access-Control-Allow-Credentials" not in resp.headers
