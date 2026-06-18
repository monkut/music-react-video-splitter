"""Google OAuth 2.0 authentication for the sanji service (issue #6).

Authorization code flow:
  GET /auth/google          → redirect to Google consent screen
  GET /auth/google/callback → exchange code, verify id_token, upsert user,
                              set session cookie
  GET /auth/logout          → clear session
"""

import os
from functools import wraps
from typing import Any, Callable

import google.oauth2.id_token
import google.auth.transport.requests
from flask import abort, redirect, session
from google_auth_oauthlib.flow import Flow
from pydantic import BaseModel

from sanji.service.users import UserStore

# ---------------------------------------------------------------------------
# Config constants (resolved from env at call time, not module import time)
# ---------------------------------------------------------------------------
GOOGLE_CLIENT_ID_ENV = "GOOGLE_CLIENT_ID"
GOOGLE_CLIENT_SECRET_ENV = "GOOGLE_CLIENT_SECRET"
OAUTH_REDIRECT_URI_ENV = "OAUTH_REDIRECT_URI"

OAUTH_SCOPES = ["openid", "email", "profile"]

# Flask session key used to store the authenticated user_id.
SESSION_USER_ID = "user_id"
# Flask session key used for CSRF state verification.
SESSION_OAUTH_STATE = "oauth_state"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CurrentUser(BaseModel):
    user_id: str
    google_sub: str
    email: str
    display_name: str
    current_plan_code: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_oauth_flow() -> Flow:
    """Build a google_auth_oauthlib Flow from env-sourced credentials."""
    client_id = os.getenv(GOOGLE_CLIENT_ID_ENV, "")
    client_secret = os.getenv(GOOGLE_CLIENT_SECRET_ENV, "")
    redirect_uri = os.getenv(
        OAUTH_REDIRECT_URI_ENV, "http://localhost:5000/auth/google/callback"
    )

    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=OAUTH_SCOPES)
    flow.redirect_uri = redirect_uri
    return flow


# ---------------------------------------------------------------------------
# Public auth functions (registered as routes in app.py)
# ---------------------------------------------------------------------------


def get_current_user() -> CurrentUser:
    """Resolve the authenticated user for this request from the session cookie."""
    user_id: str | None = session.get(SESSION_USER_ID)
    if not user_id:
        abort(401, description="Not authenticated.")

    store = UserStore()
    user = store.get(user_id)
    if user is None:
        session.pop(SESSION_USER_ID, None)
        abort(401, description="User not found.")

    return CurrentUser(
        user_id=user.user_id,
        google_sub=user.google_sub,
        email=user.email,
        display_name=user.display_name,
        current_plan_code=user.current_plan_code,
    )


def handle_google_login():
    """Redirect the browser to the Google OAuth consent screen."""
    flow = _build_oauth_flow()
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="select_account",
    )
    session[SESSION_OAUTH_STATE] = state
    return redirect(authorization_url)


def handle_google_callback(req: Any):
    """Handle the OAuth callback: verify state, exchange code, upsert user."""
    # CSRF: state must match what we stored before redirect.
    stored_state: str | None = session.pop(SESSION_OAUTH_STATE, None)
    returned_state: str | None = req.args.get("state")
    if not stored_state or stored_state != returned_state:
        abort(400, description="OAuth state mismatch — possible CSRF.")

    flow = _build_oauth_flow()
    flow.fetch_token(authorization_response=req.url)

    # Verify id_token against Google's public keys.
    credentials = flow.credentials
    id_info = google.oauth2.id_token.verify_oauth2_token(
        credentials.id_token,
        google.auth.transport.requests.Request(),
        os.getenv(GOOGLE_CLIENT_ID_ENV, ""),
    )

    google_sub: str = id_info["sub"]
    email: str = id_info.get("email", "")
    display_name: str = id_info.get("name", email)
    avatar_url: str | None = id_info.get("picture")

    # Upsert: look up by google_sub; create on first sign-in.
    store = UserStore()
    user = store.get_by_google_sub(google_sub)
    if user is None:
        user = store.create(
            google_sub=google_sub,
            email=email,
            display_name=display_name,
            avatar_url=avatar_url,
        )

    session[SESSION_USER_ID] = user.user_id
    return redirect("/")


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def login_required(view: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        user = get_current_user()
        return view(*args, current_user=user, **kwargs)

    return wrapped
