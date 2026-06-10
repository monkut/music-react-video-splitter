"""Static subscription plan configuration (D16: plans live in code, not a database).

Tier values decided 2026-06-10 — see issue #5.
Stripe Price IDs are filled in by the billing integration (#4).
"""

from pydantic import BaseModel

UNLIMITED = None


class Plan(BaseModel):
    code: str
    name: str
    price_cents: int
    monthly_stream_limit: int | None  # None = unlimited
    max_video_duration_seconds: int
    stripe_price_id: str | None = None
    active: bool = True


PLANS: tuple[Plan, ...] = (
    Plan(
        code="free",
        name="Free",
        price_cents=0,
        monthly_stream_limit=2,
        max_video_duration_seconds=2 * 60 * 60,
    ),
    Plan(
        code="pro",
        name="Pro",
        price_cents=1900,
        monthly_stream_limit=10,
        max_video_duration_seconds=4 * 60 * 60,
    ),
    Plan(
        code="business",
        name="Business",
        price_cents=4900,
        monthly_stream_limit=UNLIMITED,
        max_video_duration_seconds=8 * 60 * 60,
    ),
)

DEFAULT_PLAN_CODE = "free"


def get_plan(code: str) -> Plan | None:
    return next((plan for plan in PLANS if plan.code == code and plan.active), None)
