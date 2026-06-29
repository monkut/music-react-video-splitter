"""Tests for the completion handlers (#8).

Exercise the system-of-record side against a moto-backed sandjig Job table:
the S3-event handler finalizes a Job from ``result.json``, the Batch safety net
marks crashed Jobs failed, and both are idempotent.
"""

import boto3
import pytest
from moto import mock_aws

from sandjig.jobsapi.dyanmodb.models import ProcessingJobModel

from sanji.service.jobs import STATUS_COMPLETED, STATUS_ERROR, SanjiJobResult
from sanji.service.results import handle_batch_state_change, handle_result_event

BUCKET = "sanji-results-test"
JOB_ID = "job-xyz-789"
RESULT_KEY = f"results/{JOB_ID}/result.json"


@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")


@pytest.fixture
def aws():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )
        # Reset pynamodb's cached connection so the table binds to this mock.
        ProcessingJobModel._connection = None
        ProcessingJobModel.create_table(
            read_capacity_units=1, write_capacity_units=1, wait=True
        )
        yield s3


def _seed_job(status="queued"):
    job = ProcessingJobModel(
        job_id=JOB_ID,
        request_payload={"input_url": "https://youtu.be/x"},
        status=status,
    )
    job.save()
    return job


def _put_result_marker(s3, *, status=STATUS_COMPLETED, error=None):
    result = SanjiJobResult(
        job_id=JOB_ID,
        status=status,
        result_manifest_key=f"results/{JOB_ID}/manifest.csv",
        segment_keys=[f"results/{JOB_ID}/segments/segment_00.mp4"],
        segment_count=1,
        duration=212.5,
        error=error,
    )
    s3.put_object(Bucket=BUCKET, Key=RESULT_KEY, Body=result.model_dump_json().encode())


def _s3_event():
    return {
        "Records": [{"s3": {"bucket": {"name": BUCKET}, "object": {"key": RESULT_KEY}}}]
    }


def test_result_event_finalizes_job(aws):
    _seed_job()
    _put_result_marker(aws)

    outcome = handle_result_event(_s3_event())

    assert outcome["processed"] == 1
    job = ProcessingJobModel.get_processingjobmodel_item(JOB_ID, as_dict=False)
    assert job.status == STATUS_COMPLETED
    assert job.response_payload["segment_count"] == 1
    assert job.completed_timestamp is not None


def test_result_event_is_idempotent(aws):
    _seed_job()
    _put_result_marker(aws)

    first = handle_result_event(_s3_event())
    second = handle_result_event(_s3_event())

    assert first["processed"] == 1
    assert second["processed"] == 0  # already terminal
    job = ProcessingJobModel.get_processingjobmodel_item(JOB_ID, as_dict=False)
    assert job.status == STATUS_COMPLETED


def test_result_event_error_marker_sets_error_status(aws):
    _seed_job()
    _put_result_marker(aws, status=STATUS_ERROR, error="download failed")

    handle_result_event(_s3_event())

    job = ProcessingJobModel.get_processingjobmodel_item(JOB_ID, as_dict=False)
    assert job.status == STATUS_ERROR
    assert job.errors == {"message": "download failed"}


def test_result_event_missing_job_is_noop(aws):
    _put_result_marker(aws)  # no job seeded
    outcome = handle_result_event(_s3_event())
    assert outcome["processed"] == 0


def _batch_failed_event(*, job_id=JOB_ID, via="parameters"):
    detail = {
        "status": "FAILED",
        "jobId": "batch-internal-id",
        "statusReason": "Host EC2 instance terminated",
    }
    if via == "parameters":
        detail["parameters"] = {"job_id": job_id}
    else:
        detail["container"] = {
            "environment": [
                {"name": "SANJI_JOB_MESSAGE", "value": f'{{"job_id": "{job_id}"}}'}
            ]
        }
    return {"detail": detail}


def test_batch_safety_net_marks_failed(aws):
    _seed_job()
    outcome = handle_batch_state_change(_batch_failed_event())
    assert outcome["processed"] == 1
    job = ProcessingJobModel.get_processingjobmodel_item(JOB_ID, as_dict=False)
    assert job.status == STATUS_ERROR


def test_batch_safety_net_reads_job_id_from_container_env(aws):
    _seed_job()
    outcome = handle_batch_state_change(_batch_failed_event(via="container"))
    assert outcome["processed"] == 1


def test_batch_safety_net_does_not_override_completed(aws):
    _seed_job(status=STATUS_COMPLETED)
    outcome = handle_batch_state_change(_batch_failed_event())
    assert outcome["processed"] == 0  # result.json already won
    job = ProcessingJobModel.get_processingjobmodel_item(JOB_ID, as_dict=False)
    assert job.status == STATUS_COMPLETED


def test_batch_state_change_ignores_non_failed(aws):
    _seed_job()
    outcome = handle_batch_state_change(
        {"detail": {"status": "SUCCEEDED", "parameters": {"job_id": JOB_ID}}}
    )
    assert outcome["processed"] == 0
