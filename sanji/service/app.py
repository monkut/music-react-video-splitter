"""Flask application factory for the sanji service tier.

The app is built USING sandjig (#12/#13): ``sandjig.create_app()`` provides the
DynamoDB-backed ``/jobs`` API (+ ``/healthcheck``, ``/openapi``) and publishes
accepted jobs to the configured SQS queue; sanji's own routes are registered on
the returned Flask app afterward (extend pattern — no sandjig modifications).

Lambda entrypoint lives in ``sanji.service.lambda_app``. Since sandjig#15,
``create_app`` performs no AWS access at construction (DynamoDB init is
deferred to the first request), so importing this module is side-effect free.
"""

import os
from datetime import UTC, datetime
from typing import Any, cast

import boto3
import structlog
from flask import Flask, Response, g, jsonify, request, session
from pydantic import BaseModel
from werkzeug.exceptions import HTTPException

from sandjig.jobsapi.dyanmodb.models import ItemDoesNotExistError, ProcessingJobModel

from sanji.service.auth import (
    SESSION_USER_ID,
    CurrentUser,
    handle_google_callback,
    handle_google_login,
    login_required,
)
from sanji.service.billing import (
    BillingService,
    handle_checkout,
    handle_stripe_webhook,
)
from sanji.service.jobs import SanjiJobRequest, SanjiJobResult
from sanji.service.logging_config import configure_logging
from sanji.service.plans import DEFAULT_PLAN_CODE, PLANS, get_plan
from sanji.service.usage import UsageStore
from sanji.service.users import UserStore
from sanji.settings import (
    PRESIGN_EXPIRY_SECONDS,
    RESULTS_BUCKET_ENV,
    validate_secret_key_env_var,
    validate_stripe_env_vars,
)

# PrintLoggerFactory has no stdlib logger name; bind the required `logger` field explicitly.
logger = structlog.get_logger().bind(logger=__name__)

API_VERSION = "0.2.0"


class HealthResponse(BaseModel):
    status: str
    version: str


def create_app(config_overrides: dict[str, Any] | None = None) -> Flask:
    """Build the sandjig-based jobs app and extend it with sanji routes.

    ``config_overrides`` merges into the sandjig ``create_app`` config (used by
    tests to point SQS_QUEUE_URL at a mocked queue; production reads
    PROCESSINGJOB_REQUEST_QUEUE_URL and table names from the environment).
    """
    from sandjig import create_app as create_sandjig_app

    configure_logging(
        json_output=os.getenv("SANJI_ENVIRONMENT", "development") != "development"
    )
    validate_stripe_env_vars()
    # validate before building the sandjig app so a missing key fails fast,
    # before any AWS access
    secret_key = validate_secret_key_env_var()

    # Constructed once per app (i.e. once per Lambda cold start) and closed over
    # by the hooks and handlers below: each boto3 client/resource construction
    # rebuilds the session and credential chain, which is measurable per-request
    # latency (#40).
    user_store = UserStore()
    usage_store = UsageStore()
    billing_service = BillingService(user_store=user_store)
    s3_client = boto3.client("s3")

    def authorize_job_request(payload: dict) -> tuple[Any, int] | None:
        """sandjig JOBREQUEST_AUTHORIZATION_FUNCTION for ``POST /jobs`` (#30, #43).

        Authentication: anonymous submission is rejected with 401 — job
        submission consumes metered compute and must be attributable.

        Enforcement A: returns 402 when the authenticated user's monthly
        stream count has reached their plan cap.

        On success the resolved user/plan are stashed on ``flask.g`` for
        ``transform_job_request`` so the lookups run once per request.
        """
        user_id: str | None = session.get(SESSION_USER_ID)
        if not user_id:
            return jsonify(
                error="unauthorized", message="Authentication required."
            ), 401

        user = user_store.get(user_id)
        if user is None:
            session.pop(SESSION_USER_ID, None)
            return jsonify(
                error="unauthorized", message="Authentication required."
            ), 401

        plan = get_plan(user.current_plan_code)
        if plan is None:
            logger.warning(
                "unknown_plan_code", user_id=user_id, plan=user.current_plan_code
            )
            plan = get_plan(DEFAULT_PLAN_CODE)

        period_key = datetime.now(UTC).strftime("%Y-%m")
        current_count = usage_store.get_monthly_count(user_id, period_key)
        if current_count >= plan.monthly_stream_limit:
            logger.info(
                "plan_limit_reached",
                user_id=user_id,
                plan=user.current_plan_code,
                count=current_count,
                limit=plan.monthly_stream_limit,
            )
            return (
                jsonify(
                    error="plan_limit_exceeded",
                    message=(
                        f"Monthly stream limit of {plan.monthly_stream_limit} reached"
                        f" for plan '{plan.code}'."
                    ),
                    limit=plan.monthly_stream_limit,
                    current_count=current_count,
                ),
                402,
            )

        g.job_request_user_id = user_id
        g.job_request_plan = plan
        return None

    def transform_job_request(payload: dict) -> dict:
        """sandjig JOBREQUEST_TRANSFORM_FUNCTION: inject server-side fields (#43).

        Overwrites any client-supplied ``user_id`` with the session user (#30)
        and adds max_duration_seconds so the async worker enforces the per-video
        duration cap (enforcement B) without its own plan lookup. Runs after
        ``authorize_job_request``, which resolved the user/plan onto ``flask.g``.
        """
        params = dict(payload.get("params") or {})
        params.setdefault(
            "max_duration_seconds", g.job_request_plan.max_video_duration_seconds
        )
        payload["params"] = params
        payload["user_id"] = g.job_request_user_id
        return payload

    config: dict[str, Any] = {
        "API_TITLE": "sanji API",
        "API_VERSION": API_VERSION,
        # Supported sandjig request hooks (sandjig#14) — replaces the private
        # request._cached_json mutation this app previously relied on (#43).
        "JOBREQUEST_AUTHORIZATION_FUNCTION": authorize_job_request,
        "JOBREQUEST_TRANSFORM_FUNCTION": transform_job_request,
    }
    if config_overrides:
        config.update(config_overrides)

    app = create_sandjig_app(SanjiJobRequest, SanjiJobResult, config=config)

    # Signed session cookie configuration.
    app.secret_key = secret_key
    is_production = os.getenv("SANJI_ENVIRONMENT", "development") != "development"
    app.config["SESSION_COOKIE_SECURE"] = is_production
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    @app.after_request
    def handle_usage_increment(response: Response) -> Response:
        """Count a successful job submission against the user's monthly cap (#32).

        Runs only for ``POST /jobs`` 201 responses; failed submissions (401/402/
        422/5xx) do not consume quota. The increment uses the atomic DynamoDB
        ADD in UsageStore, so concurrent submissions cannot double-count.
        """
        if (
            request.method == "POST"
            and request.path == "/jobs"
            and response.status_code == 201
        ):
            user_id: str | None = session.get(SESSION_USER_ID)
            if user_id:
                try:
                    count = usage_store.increment_current_count(user_id)
                    logger.info("usage_incremented", user_id=user_id, count=count)
                except Exception as exc:
                    # the job is already enqueued — do not fail the response;
                    # an uncounted job under-bills, a failed response double-runs
                    logger.error(
                        "usage_increment_failed", user_id=user_id, error=str(exc)
                    )
        return response

    @app.get("/health")
    def health() -> tuple[dict, int]:
        return HealthResponse(status="ok", version=API_VERSION).model_dump(), 200

    @app.get("/plans")
    def plans() -> tuple[dict, int]:
        return {"plans": [plan.model_dump() for plan in PLANS]}, 200

    @app.get("/me")
    @login_required
    def me(current_user: CurrentUser) -> tuple[dict, int]:
        return current_user.model_dump(), 200

    @app.get("/jobs/<job_id>/result")
    @login_required
    def job_result(job_id: str, current_user: CurrentUser) -> tuple[dict, int]:
        """Job status plus presigned download URLs for its result artifacts.

        sandjig owns ``GET /jobs/<id>`` (status + raw keys); this companion route
        turns the stored S3 keys into time-limited download URLs (#8).

        Requires login; only the job owner may fetch result URLs (#30) — a
        non-owner receives 404 so job ids are not confirmable by probing.
        """
        try:
            job = cast(
                dict,
                ProcessingJobModel.get_processingjobmodel_item(job_id, as_dict=True),
            )
        except ItemDoesNotExistError:
            return {"error": "not_found"}, 404

        owner_id = (job.get("request_payload") or {}).get("user_id")
        if owner_id != current_user.user_id:
            logger.info(
                "job_result_ownership_denied",
                job_id=job_id,
                user_id=current_user.user_id,
            )
            return {"error": "not_found"}, 404

        payload = job.get("response_payload") or {}
        bucket = os.getenv(RESULTS_BUCKET_ENV)
        if not bucket:
            logger.warning("results_bucket_unset", env=RESULTS_BUCKET_ENV)

        def presign(key: str) -> str:
            return s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=PRESIGN_EXPIRY_SECONDS,
            )

        manifest_key = payload.get("result_manifest_key")
        body = {
            "job_id": job_id,
            "status": job.get("status"),
            "segment_urls": [presign(key) for key in payload.get("segment_keys", [])]
            if bucket
            else [],
            "manifest_url": presign(manifest_key)
            if (bucket and manifest_key)
            else None,
        }
        return body, 200

    @app.get("/auth/google")
    def google_login():
        return handle_google_login()

    @app.get("/auth/google/callback")
    def google_callback():
        return handle_google_callback(request)

    @app.post("/billing/checkout")
    @login_required
    def billing_checkout(current_user: CurrentUser) -> tuple[dict, int]:
        return handle_checkout(current_user, billing_service)

    @app.post("/webhooks/stripe")
    def stripe_webhook() -> tuple[dict, int]:
        return handle_stripe_webhook(billing_service)

    @app.post("/auth/logout")
    def logout() -> tuple[str, int]:
        from flask import session as flask_session

        flask_session.clear()
        return "", 204

    # Later registration wins: these replace sandjig's plain-text handlers so
    # the host app owns error semantics (verified extend-pattern behavior).
    @app.errorhandler(401)
    def unauthorized(error: HTTPException) -> tuple[Response, int]:
        return jsonify(error="unauthorized", message=str(error.description or "")), 401

    @app.errorhandler(404)
    def not_found(error: HTTPException) -> tuple[Response, int]:
        return jsonify(error="not_found"), 404

    @app.errorhandler(500)
    def server_error(error: HTTPException) -> tuple[Response, int]:
        logger.error("unhandled_error", error=str(error))
        return jsonify(error="internal_server_error"), 500

    logger.info("app_created", version=API_VERSION)
    return app
