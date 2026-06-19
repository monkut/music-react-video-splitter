"""Tests for the AWS Batch worker (#8).

The pipeline itself is mocked — these verify the worker's S3 contract: segments
and manifest land under ``results/<job_id>/``, ``result.json`` is written last as
the completion marker, and a pipeline failure still produces a terminal marker.
"""

import json

import boto3
import pytest
from moto import mock_aws

from sanji.pipeline import PipelineResult
from sanji.service.jobs import STATUS_COMPLETED, STATUS_ERROR, SanjiJobRequest
from sanji.worker import (
    RESULTS_ROOT,
    build_pipeline_params,
    parse_job_message,
    process_job,
)

BUCKET = "sanji-results-test"
JOB_ID = "job-abc-123"


@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-west-2")
        client.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )
        yield client


def _fake_pipeline_result(tmp_path, *, segments=2):
    segment_files = []
    for i in range(segments):
        seg = tmp_path / f"segment_{i:02d}.mp4"
        seg.write_bytes(f"video-bytes-{i}".encode())
        segment_files.append(seg)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("start,end,track\n0,60,one\n")
    return PipelineResult(
        duration=212.5,
        song_regions=[(0.0, 60.0)],
        split_points=[60.0],
        track_names=["one"],
        segment_files=segment_files,
        manifest_path=manifest,
    )


def _list_keys(s3):
    response = s3.list_objects_v2(Bucket=BUCKET)
    return sorted(obj["Key"] for obj in response.get("Contents", []))


def test_parse_job_message_extracts_job_id_and_request():
    raw = json.dumps(
        {
            "job_id": JOB_ID,
            "request_payload": {
                "input_url": "https://youtu.be/x",
                "params": {"threshold": 0.4},
            },
        }
    )
    job_id, request = parse_job_message(raw)
    assert job_id == JOB_ID
    assert isinstance(request, SanjiJobRequest)
    assert request.input_url == "https://youtu.be/x"
    assert request.params == {"threshold": 0.4}


def test_parse_job_message_requires_job_id():
    with pytest.raises(ValueError, match="job_id"):
        parse_job_message(
            json.dumps({"request_payload": {"input_url": "https://youtu.be/x"}})
        )


def test_build_pipeline_params_filters_unknown_overrides(tmp_path):
    request = SanjiJobRequest(
        input_url="https://youtu.be/x", params={"threshold": 0.7, "bogus": 1}
    )
    params = build_pipeline_params(request, tmp_path / "out")
    assert params.input == "https://youtu.be/x"
    assert params.threshold == 0.7
    assert not hasattr(params, "bogus")


def test_process_job_uploads_artifacts_and_marker(s3, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sanji.worker.run_pipeline", lambda *a, **k: _fake_pipeline_result(tmp_path)
    )
    request = SanjiJobRequest(input_url="https://youtu.be/x", params={})

    result = process_job(JOB_ID, request, BUCKET, s3_client=s3)

    keys = _list_keys(s3)
    prefix = f"{RESULTS_ROOT}/{JOB_ID}"
    assert f"{prefix}/segments/segment_00.mp4" in keys
    assert f"{prefix}/segments/segment_01.mp4" in keys
    assert f"{prefix}/manifest.csv" in keys
    assert f"{prefix}/result.json" in keys

    assert result.status == STATUS_COMPLETED
    assert result.job_id == JOB_ID
    assert result.segment_count == 2
    assert result.duration == 212.5
    assert result.result_manifest_key == f"{prefix}/manifest.csv"


def test_result_marker_references_uploaded_segments(s3, tmp_path, monkeypatch):
    """result.json carries the segment keys — proof it is written after the uploads."""
    monkeypatch.setattr(
        "sanji.worker.run_pipeline", lambda *a, **k: _fake_pipeline_result(tmp_path)
    )
    request = SanjiJobRequest(input_url="https://youtu.be/x")

    process_job(JOB_ID, request, BUCKET, s3_client=s3)

    marker = s3.get_object(Bucket=BUCKET, Key=f"{RESULTS_ROOT}/{JOB_ID}/result.json")[
        "Body"
    ].read()
    payload = json.loads(marker)
    assert payload["status"] == STATUS_COMPLETED
    assert payload["segment_keys"] == [
        f"{RESULTS_ROOT}/{JOB_ID}/segments/segment_00.mp4",
        f"{RESULTS_ROOT}/{JOB_ID}/segments/segment_01.mp4",
    ]


def test_process_job_writes_error_marker_on_failure(s3, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr("sanji.worker.run_pipeline", boom)
    request = SanjiJobRequest(input_url="https://youtu.be/x")

    result = process_job(JOB_ID, request, BUCKET, s3_client=s3)

    assert result.status == STATUS_ERROR
    assert result.error == "download failed"
    keys = _list_keys(s3)
    assert keys == [f"{RESULTS_ROOT}/{JOB_ID}/result.json"]  # no segments on failure
