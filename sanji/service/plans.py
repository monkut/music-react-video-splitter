"""Static subscription plan configuration (D16: plans live in code, not a database).

Tier values decided 2026-06-10; updated 2026-07-01 — see issue #5.
Stripe Price IDs are filled in by the billing integration (#4).
"""

from pydantic import BaseModel

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
