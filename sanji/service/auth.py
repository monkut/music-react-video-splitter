"""Authentication seam (filled in by #6 Google OAuth).

Until #6 lands, any endpoint requiring a user returns 401.
"""

from functools import wraps
from typing import Any, Callable

from flask import abort
from pydantic import BaseModel


class CurrentUser(BaseModel):
    user_id: str
    google_sub: str
    email: str
    display_name: str
    current_plan_code: str


def get_current_user() -> CurrentUser:
    """Resolve the authenticated user for this request.

    Stub: always 401 until #6 implements Google OAuth (JWT in httpOnly cookie, D1).
    """
    abort(401, description="Authentication not yet available (see issue #6).")


def login_required(view: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        user = get_current_user()
        return view(*args, current_user=user, **kwargs)

    return wrapped
