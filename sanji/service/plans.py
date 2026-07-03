"""Static subscription plan configuration (D16: plans live in code, not a database).

Tier values decided 2026-06-10; updated 2026-07-01 — see issue #5.
Stripe Price IDs are deploy-specific and resolved from the environment at
lookup time via ``resolve_stripe_price_id`` (#37).
"""

import os

from pydantic import BaseModel

# Env var per paid plan holding that deploy's Stripe Price ID (#37).
STRIPE_PRICE_ENV_VARS: dict[str, str] = {
    "pro": "SANJI_STRIPE_PRICE_PRO",
    "business": "SANJI_STRIPE_PRICE_BUSINESS",
}

MAX_DURATION_SECONDS = 2 * 60 * 60  # 2 hours — shared cap across all tiers


class Plan(BaseModel):
    code: str
    name: str
    price_cents: int
    monthly_stream_limit: int  # hard cap on all tiers
    max_video_duration_seconds: int
    stripe_price_id: str | None = None
    active: bool = True


PLANS: tuple[Plan, ...] = (
    Plan(
        code="free",
        name="Free",
        price_cents=0,
        monthly_stream_limit=2,
        max_video_duration_seconds=MAX_DURATION_SECONDS,
    ),
    Plan(
        code="pro",
        name="Pro",
        price_cents=1900,
        monthly_stream_limit=10,
        max_video_duration_seconds=MAX_DURATION_SECONDS,
    ),
    Plan(
        code="business",
        name="Business",
        price_cents=3900,
        monthly_stream_limit=30,
        max_video_duration_seconds=MAX_DURATION_SECONDS,
    ),
)

DEFAULT_PLAN_CODE = "free"


def get_plan(code: str) -> Plan | None:
    return next((plan for plan in PLANS if plan.code == code and plan.active), None)


def resolve_stripe_price_id(plan: Plan) -> str | None:
    """Return the plan's Stripe Price ID, preferring the deploy's env value.

    Resolved at lookup time (not import time) so the Lambda environment and
    tests control the mapping without mutating the static PLANS instances.
    """
    env_var = STRIPE_PRICE_ENV_VARS.get(plan.code)
    env_value = os.getenv(env_var) if env_var else None
    return env_value or plan.stripe_price_id
