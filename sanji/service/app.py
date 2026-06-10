"""Flask application for the sanji service tier.

Deployed to AWS Lambda via Zappa (``app`` is the Zappa ``app_function`` target).
"""

import os

import structlog
from flask import Flask, Response, jsonify
from pydantic import BaseModel
from werkzeug.exceptions import HTTPException

from sanji.service.auth import CurrentUser, login_required
from sanji.service.logging_config import configure_logging
from sanji.service.plans import PLANS

# PrintLoggerFactory has no stdlib logger name; bind the required `logger` field explicitly.
logger = structlog.get_logger().bind(logger=__name__)

API_VERSION = "0.1.0"


class HealthResponse(BaseModel):
    status: str
    version: str


def create_app() -> Flask:
    configure_logging(
        json_output=os.getenv("SANJI_ENVIRONMENT", "development") != "development"
    )
    app = Flask("sanji-api")

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


app = create_app()
