"""Completion handlers — the system-of-record side of the async pipeline (#8).

Two Lambda entry points close the loop the worker opens:

* ``handle_result_event`` — fires on ``result.json`` ObjectCreated. Loads the
  worker's ``SanjiJobResult`` and records the terminal status + payload on the
  Job. This is the happy path (and graceful failures, which also write a marker).
* ``handle_batch_state_change`` — the safety net. Fires on a Batch Job State
  Change to ``FAILED``; marks the Job ``error`` when the worker died without
  writing ``result.json`` (Spot reclaim, OOM, timeout).

Both updates are idempotent: a conditional write guards on the Job not already
being in a terminal state, so a redelivered S3 event or a late Batch failure
after a successful run is a no-op.
"""

from __future__ import annotations

import json
from typing import Any, cast
from urllib.parse import unquote_plus

import boto3
import structlog
from pynamodb.exceptions import UpdateError

from sandjig.functions import get_timestamp_now
from sandjig.jobsapi.dynamodb.models import ItemDoesNotExistError, ProcessingJobModel

from sanji.service.jobs import STATUS_COMPLETED, STATUS_ERROR, SanjiJobResult

logger = structlog.get_logger().bind(logger=__name__)

TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_ERROR)
BATCH_FAILED_STATUS = "FAILED"


def _record_terminal_status(
    job_id: str,
    status: str,
    *,
    response_payload: dict | None = None,
    error: str | None = None,
) -> bool:
    """Idempotently move a Job to a terminal state. Returns True if this call applied it.

    The conditional update guards on the Job existing and not already being
    terminal, so concurrent or redelivered events resolve to a single winner.
    """
    try:
        job = cast(
            ProcessingJobModel,
            ProcessingJobModel.get_processingjobmodel_item(job_id, as_dict=False),
        )
    except ItemDoesNotExistError:
        logger.warning("terminal_status_job_missing", job_id=job_id, status=status)
        return False

    if job.status in TERMINAL_STATUSES:
        logger.info("terminal_status_noop", job_id=job_id, current_status=job.status)
        return False

    now = get_timestamp_now()
    actions: list[Any] = [
        ProcessingJobModel.status.set(status),
        ProcessingJobModel.updated_timestamp.set(now),
        ProcessingJobModel.completed_timestamp.set(now),
    ]
    if response_payload is not None:
        actions.append(ProcessingJobModel.response_payload.set(response_payload))
    if error is not None:
        actions.append(ProcessingJobModel.errors.set({"message": error}))

    try:
        job.update(
            actions=actions,
            condition=ProcessingJobModel.job_id.exists()
            & ~ProcessingJobModel.status.is_in(*TERMINAL_STATUSES),
        )
    except UpdateError as exc:
        # Lost the race — another invocation already finalized this Job.
        logger.info(
            "terminal_status_condition_failed",
            job_id=job_id,
            status=status,
            detail=str(exc),
        )
        return False

    logger.info("terminal_status_applied", job_id=job_id, status=status)
    return True


def _read_result(bucket: str, key: str, *, s3_client: Any = None) -> SanjiJobResult:
    s3 = s3_client or boto3.client("s3")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return SanjiJobResult.model_validate_json(body)


def handle_result_event(
    event: dict, context: Any = None, *, s3_client: Any = None
) -> dict:
    """S3 ObjectCreated handler for ``result.json`` completion markers."""
    processed = 0
    for record in event.get("Records", []):
        s3_info = record["s3"]
        bucket = s3_info["bucket"]["name"]
        key = unquote_plus(s3_info["object"]["key"])
        result = _read_result(bucket, key, s3_client=s3_client)
        job_id = result.job_id
        if not job_id:
            logger.warning("result_marker_missing_job_id", bucket=bucket, key=key)
            continue
        if _record_terminal_status(
            job_id,
            result.status,
            response_payload=result.model_dump(),
            error=result.error,
        ):
            processed += 1
    return {"processed": processed}


def _extract_job_id(detail: dict) -> str | None:
    """Pull the sanji job_id out of a Batch Job State Change detail.

    Prefer the explicit ``job_id`` Batch parameter; fall back to parsing it from
    the worker's job message in the container environment.
    """
    job_id = (detail.get("parameters") or {}).get("job_id")
    if job_id:
        return job_id
    container = detail.get("container") or {}
    for env in container.get("environment", []):
        if env.get("name") == "SANJI_JOB_MESSAGE":
            try:
                return json.loads(env["value"]).get("job_id")
            except (json.JSONDecodeError, KeyError):
                return None
    return None


def handle_batch_state_change(event: dict, context: Any = None) -> dict:
    """EventBridge safety net: mark Jobs failed when Batch terminates without a marker."""
    detail = event.get("detail", {})
    if detail.get("status") != BATCH_FAILED_STATUS:
        logger.debug("batch_state_change_ignored", status=detail.get("status"))
        return {"processed": 0}

    job_id = _extract_job_id(detail)
    if not job_id:
        logger.warning("batch_failure_missing_job_id", batch_job_id=detail.get("jobId"))
        return {"processed": 0}

    reason = detail.get("statusReason") or "batch job terminated without result.json"
    applied = _record_terminal_status(job_id, STATUS_ERROR, error=reason)
    return {"processed": int(applied)}
