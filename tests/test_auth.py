"""Tests for Google OAuth 2.0 authentication (issue #6).

AWS (DynamoDB, SQS) is mocked with moto. The Google OAuth library calls are
patched so tests run offline without real credentials.
"""

from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

from sanji.service.users import GOOGLE_SUB_INDEX, UserStore


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")


def _create_users_table(dynamodb_client):
    dynamodb_client.create_table(
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


@pytest.fixture
def app_client():
    """Flask test client with mocked AWS and a test secret key."""
    with mock_aws():
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
        _create_users_table(dynamodb)

        sqs = boto3.client("sqs", region_name="us-west-2")
        queue_url = sqs.create_queue(QueueName="sanji-jobs-test")["QueueUrl"]

        from sanji.service.app import create_app

        app = create_app(config_overrides={"SQS_QUEUE_URL": queue_url})
        app.config["TESTING"] = True
        app.secret_key = "test-secret"
        yield app.test_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_ID_INFO = {
    "sub": "google-sub-001",
    "email": "alice@example.com",
    "name": "Alice Example",
    "picture": "https://example.com/avatar.png",
}


def _mock_flow_and_verify(monkeypatch, id_info: dict | None = None):
    """Patch Flow.fetch_token and verify_oauth2_token for a callback test."""
    info = id_info or _FAKE_ID_INFO

    mock_credentials = MagicMock()
    mock_credentials.id_token = "fake-id-token"

    mock_flow = MagicMock()
    mock_flow.credentials = mock_credentials

    monkeypatch.setattr(
        "sanji.service.auth._build_oauth_flow",
        lambda: mock_flow,
    )
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda token, request, audience: info,
    )
    return mock_flow


# ---------------------------------------------------------------------------
# /me endpoint (unauthenticated / authenticated)
# ---------------------------------------------------------------------------


def test_me_unauthenticated_returns_401(app_client):
    response = app_client.get("/me")
    assert response.status_code == 401
    assert response.json["error"] == "unauthorized"


def test_me_authenticated_returns_user(app_client, monkeypatch):
    """Seed a user directly in DynamoDB and set the session manually."""
    _mock_flow_and_verify(monkeypatch)

    with app_client.session_transaction() as sess:
        # Create the user directly, then inject the user_id into the session.
        store = UserStore()
        user = store.create(
            google_sub="google-sub-me",
            email="me@example.com",
            display_name="Me User",
        )
        sess["user_id"] = user.user_id

    response = app_client.get("/me")
    assert response.status_code == 200
    assert response.json["email"] == "me@example.com"
    assert response.json["display_name"] == "Me User"


# ---------------------------------------------------------------------------
# /auth/google → redirect
# ---------------------------------------------------------------------------


def test_google_login_redirects_to_google(app_client, monkeypatch):
    mock_flow = MagicMock()
    mock_flow.authorization_url.return_value = (
        "https://accounts.google.com/o/oauth2/auth?response_type=code",
        "state-abc",
    )
    monkeypatch.setattr("sanji.service.auth._build_oauth_flow", lambda: mock_flow)

    response = app_client.get("/auth/google")
    assert response.status_code == 302
    assert "accounts.google.com" in response.headers["Location"]


# ---------------------------------------------------------------------------
# /auth/google/callback
# ---------------------------------------------------------------------------


def test_google_callback_creates_new_user(app_client, monkeypatch):
    """First sign-in with a new google_sub creates a user record."""
    _mock_flow_and_verify(monkeypatch)

    # Inject a matching oauth_state into the session.
    with app_client.session_transaction() as sess:
        sess["oauth_state"] = "state-xyz"

    response = app_client.get(
        "/auth/google/callback?state=state-xyz&code=authcode123",
        follow_redirects=False,
    )
    assert response.status_code == 302, response.data

    # Verify user was created in DynamoDB.
    store = UserStore()
    user = store.get_by_google_sub(_FAKE_ID_INFO["sub"])
    assert user is not None
    assert user.email == _FAKE_ID_INFO["email"]
    assert user.display_name == _FAKE_ID_INFO["name"]

    # Session should now carry user_id.
    with app_client.session_transaction() as sess:
        assert sess.get("user_id") == user.user_id


def test_google_callback_matches_existing_user(app_client, monkeypatch):
    """Second sign-in with same google_sub returns the existing user — no duplicate."""
    _mock_flow_and_verify(monkeypatch)

    # Pre-create the user.
    store = UserStore()
    existing = store.create(
        google_sub=_FAKE_ID_INFO["sub"],
        email=_FAKE_ID_INFO["email"],
        display_name=_FAKE_ID_INFO["name"],
    )

    with app_client.session_transaction() as sess:
        sess["oauth_state"] = "state-xyz2"

    app_client.get(
        "/auth/google/callback?state=state-xyz2&code=authcode456",
        follow_redirects=False,
    )

    # Still only one user with this sub.
    user = store.get_by_google_sub(_FAKE_ID_INFO["sub"])
    assert user is not None
    assert user.user_id == existing.user_id


def test_google_callback_rejects_state_mismatch(app_client, monkeypatch):
    """Mismatched state triggers 400 (CSRF protection)."""
    _mock_flow_and_verify(monkeypatch)

    with app_client.session_transaction() as sess:
        sess["oauth_state"] = "correct-state"

    response = app_client.get(
        "/auth/google/callback?state=wrong-state&code=authcode789",
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# /auth/logout
# ---------------------------------------------------------------------------


def test_logout_clears_session(app_client, monkeypatch):
    """GET /auth/logout clears the session and returns 204."""
    _mock_flow_and_verify(monkeypatch)

    with app_client.session_transaction() as sess:
        sess["user_id"] = "some-user-id"

    response = app_client.get("/auth/logout")
    assert response.status_code == 204

    with app_client.session_transaction() as sess:
        assert "user_id" not in sess
