"""structlog configuration per the org logging standard.

JSON output in production, console in development. Required fields:
timestamp, level, event, logger, service, environment.
"""

import os

import structlog

SERVICE_NAME = "sanji-api"

_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "token",
        "secret",
        "ssn",
        "email",
        "credit_card",
        "authorization",
        "api_key",
    }
)


def scrub_sensitive_fields(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    for key in _SENSITIVE_KEYS & event_dict.keys():
        event_dict[key] = "[REDACTED]"
    return event_dict


def _add_service_context(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    event_dict["service"] = SERVICE_NAME
    event_dict["environment"] = os.getenv("SANJI_ENVIRONMENT", "development")
    return event_dict


def configure_logging(*, json_output: bool = True) -> None:
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        scrub_sensitive_fields,
        _add_service_context,
    ]
    renderer: structlog.types.Processor
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
