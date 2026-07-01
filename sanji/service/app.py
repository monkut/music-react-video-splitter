"""Flask application factory for the sanji service tier.

The app is built USING sandjig (#12/#13): ``sandjig.create_app()`` provides the
DynamoDB-backed ``/jobs`` API (+ ``/healthcheck``, ``/openapi``) and publishes
accepted jobs to the configured SQS queue; sanji's own routes are registered on
the returned Flask app afterward (extend pattern — no sandjig modifications).

Lambda entrypoint lives in ``sanji.service.lambda_app`` (sandjig touches
DynamoDB at app construction, so importing this module stays side-effect free).
"""

import os
from datetime import UTC, datetime
from typing import Any, cast

import boto3
import structlog
from flask import Flask, Response, jsonify, request, session
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
from sanji.service.jobs import SanjiJobRequest, SanjiJobResult
from sanji.service.logging_config import configure_logging
from sanji.service.plans import PLANS, get_plan
from sanji.service.usage import UsageStore
from sanji.service.users import UserStore
from sanji.settings import PRESIGN_EXPIRY_SECONDS, RESULTS_BUCKET_ENV, validate_stripe_env_vars

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

    config: dict[str, Any] = {
        "API_TITLE": "sanji API",
        "API_VERSION": API_VERSION,
    }
    if config_overrides:
        config.update(config_overrides)

    app = create_sandjig_app(SanjiJobRequest, SanjiJobResult, config=config)

    # Signed session cookie configuration.
    app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
    is_production = os.getenv("SANJI_ENVIRONMENT", "development") != "development"
    app.config["SESSION_COOKIE_SECURE"] = is_production
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    @app.before_request
    def handle_plan_enforcement():
        """Enforce plan limits before sandjig enqueues a job.

        Enforcement A: Returns HTTP 402 when the authenticated user's monthly
        stream count has reached their plan cap.

        Duration injection: Adds max_duration_seconds to the job params so the
        async worker can enforce the per-video duration cap (enforcement B) without
        needing to look up plan data from DynamoDB itself.

        Unauthenticated requests pass through unchanged.
        """
        if request.method != "POST" or request.path != "/jobs":
            return None

        user_id: str | None = session.get(SESSION_USER_ID)
        if not user_id:
            return None

        user = UserStore().get(user_id)
        if user is None:
            return None

        plan = get_plan(user.current_plan_code)
        if plan is None:
            return None

        period_key = datetime.now(UTC).strftime("%Y-%m")
        current_count = UsageStore().get_monthly_count(user_id, period_key)
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

        # Inject max_duration_seconds into params so the worker enforces duration (B).
        # Overwrite the request's cached JSON so sandjig reads the augmented body.
        body = dict(request.get_json(silent=True, force=True) or {})
        params = dict(body.get("params") or {})
        params.setdefault("max_duration_seconds", plan.max_video_duration_seconds)
        body["params"] = params
        request._cached_json = (body, body)  # type: ignore[assignment]
        return None

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
    def job_result(job_id: str) -> tuple[dict, int]:
        """Job status plus presigned download URLs for its result artifacts.

        sandjig owns ``GET /jobs/<id>`` (status + raw keys); this companion route
        turns the stored S3 keys into time-limited download URLs (#8).
        """
        try:
            job = cast(
                dict,
                ProcessingJobModel.get_processingjobmodel_item(job_id, as_dict=True),
            )
        except ItemDoesNotExistError:
            return {"error": "not_found"}, 404

        payload = job.get("response_payload") or {}
        bucket = os.getenv(RESULTS_BUCKET_ENV)
        if not bucket:
            logger.warning("results_bucket_unset", env=RESULTS_BUCKET_ENV)

        s3 = boto3.client("s3")

        def presign(key: str) -> str:
            return s3.generate_presigned_url(
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

    @app.get("/auth/logout")
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
