"""AWS Batch worker: run the splitting pipeline and land results in S3 (#8).

Thin, fully-decoupled handler over ``run_pipeline``. It receives the sandjig
job message (``job_id`` + ``request_payload``) — the same JSON the EventBridge
Pipe forwards from the jobs SQS queue — runs the pipeline, streams the segment
files and manifest to S3, then writes ``result.json`` **last** as the completion
marker the system-of-record watches for.

Boundary (locked 2026-06-01): the worker touches **S3 only** (``s3:PutObject``).
No DynamoDB, no API, no SQS. A caught pipeline failure is still reported as a
terminal ``result.json`` (``status=error``); hard kills (Spot reclaim, OOM,
timeout) leave no marker and are handled by the Batch state-change safety net.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import boto3
import structlog

from sanji.pipeline import PipelineParams, PipelineResult, run_pipeline
from sanji.service.jobs import (
    STATUS_COMPLETED,
    STATUS_ERROR,
    SanjiJobRequest,
    SanjiJobResult,
)

logger = structlog.get_logger().bind(logger=__name__)

# The worker reads its job message from this env var (set as a Batch container
# override) or, failing that, from argv[1].
JOB_MESSAGE_ENV = "SANJI_JOB_MESSAGE"
# Destination bucket for results — provisioned by the infra stack (#11).
RESULTS_BUCKET_ENV = "SANJI_RESULTS_BUCKET"
RESULTS_ROOT = "results"
RESULT_MARKER_NAME = "result.json"


class ResultUploader:
    """Owns the S3 client + bucket and streams a run's artifacts under one prefix."""

    def __init__(self, bucket: str, prefix: str, *, s3_client: Any = None) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._s3 = s3_client or boto3.client("s3")

    def _put(self, key: str, body: bytes, content_type: str) -> str:
        self._s3.put_object(
            Bucket=self._bucket, Key=key, Body=body, ContentType=content_type
        )
        logger.info(
            "result_object_uploaded", bucket=self._bucket, key=key, bytes=len(body)
        )
        return key

    def upload_segment(self, path: Path) -> str:
        return self._put(
            f"{self._prefix}/segments/{path.name}", path.read_bytes(), "video/mp4"
        )

    def upload_manifest(self, path: Path) -> str:
        return self._put(f"{self._prefix}/manifest.csv", path.read_bytes(), "text/csv")

    def write_marker(self, result: SanjiJobResult) -> str:
        """Write ``result.json`` last — its arrival is the completion signal."""
        body = result.model_dump_json().encode("utf-8")
        return self._put(
            f"{self._prefix}/{RESULT_MARKER_NAME}", body, "application/json"
        )


def parse_job_message(raw: str) -> tuple[str, SanjiJobRequest]:
    """Extract ``job_id`` and the request payload from a sandjig job message."""
    message = json.loads(raw)
    job_id = message.get("job_id")
    if not job_id:
        raise ValueError("job message missing 'job_id'")
    payload = message.get("request_payload", message)
    return job_id, SanjiJobRequest.model_validate(payload)


def build_pipeline_params(request: SanjiJobRequest, output_dir: Path) -> PipelineParams:
    """Map the request payload onto ``PipelineParams``, honoring known overrides only."""
    tunable = {f.name for f in dataclasses.fields(PipelineParams)} - {
        "input",
        "output_dir",
    }
    overrides = {key: value for key, value in request.params.items() if key in tunable}
    return PipelineParams(input=request.input_url, output_dir=output_dir, **overrides)


def _build_result(
    job_id: str,
    prefix: str,
    segment_keys: list[str],
    manifest_key: str | None,
    pipeline_result: PipelineResult,
) -> SanjiJobResult:
    return SanjiJobResult(
        job_id=job_id,
        status=STATUS_COMPLETED,
        result_manifest_key=manifest_key,
        segment_keys=segment_keys,
        segment_count=len(segment_keys),
        duration=pipeline_result.duration,
    )


def process_job(
    job_id: str, request: SanjiJobRequest, bucket: str, *, s3_client: Any = None
) -> SanjiJobResult:
    """Run the pipeline for one job and land its artifacts in S3.

    Returns the ``SanjiJobResult`` that was written as ``result.json``. On a caught
    pipeline failure a terminal ``status=error`` marker is written and returned, so
    the system-of-record always sees a completion signal for graceful failures.
    """
    prefix = f"{RESULTS_ROOT}/{job_id}"
    uploader = ResultUploader(bucket, prefix, s3_client=s3_client)
    logger.info(
        "job_started", job_id=job_id, input_url=request.input_url, bucket=bucket
    )

    with tempfile.TemporaryDirectory(prefix=f"sanji_{job_id}_") as scratch:
        scratch_dir = Path(scratch)
        try:
            pipeline_result = run_pipeline(
                build_pipeline_params(request, scratch_dir / "output"),
                work_dir=scratch_dir / "work",
            )
        except Exception as exc:  # noqa: BLE001 — any failure becomes a terminal marker
            logger.exception("job_failed", job_id=job_id)
            error_result = SanjiJobResult(
                job_id=job_id, status=STATUS_ERROR, error=str(exc)
            )
            uploader.write_marker(error_result)
            return error_result

        segment_keys = [
            uploader.upload_segment(path) for path in pipeline_result.segment_files
        ]
        manifest_key = (
            uploader.upload_manifest(pipeline_result.manifest_path)
            if pipeline_result.manifest_path
            else None
        )
        result = _build_result(
            job_id, prefix, segment_keys, manifest_key, pipeline_result
        )
        uploader.write_marker(result)  # last: completion marker
        logger.info("job_completed", job_id=job_id, segment_count=result.segment_count)
        return result


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    raw = os.getenv(JOB_MESSAGE_ENV) or (argv[1] if len(argv) > 1 else None)
    if not raw:
        logger.error("missing_job_message", env=JOB_MESSAGE_ENV)
        return 2
    bucket = os.getenv(RESULTS_BUCKET_ENV)
    if not bucket:
        logger.error("missing_results_bucket", env=RESULTS_BUCKET_ENV)
        return 2

    job_id, request = parse_job_message(raw)
    result = process_job(job_id, request, bucket)
    return 0 if result.status == STATUS_COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
