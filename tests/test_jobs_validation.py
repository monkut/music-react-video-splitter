"""input_url validation tests (issue #33).

The validator lives on ``SanjiJobRequest`` so it applies at both entry points:
sandjig validates the API request body against the model (422), and the worker
re-validates via ``parse_job_message``.
"""

import json

import pytest
from pydantic import ValidationError

from sanji.service.jobs import SanjiJobRequest
from sanji.worker import parse_job_message

VALID_URLS = (
    "https://www.youtube.com/watch?v=TESTVIDEO01",
    "https://youtube.com/watch?v=TESTVIDEO01",
    "https://m.youtube.com/watch?v=TESTVIDEO01",
    "https://music.youtube.com/watch?v=TESTVIDEO01",
    "https://youtu.be/TESTVIDEO01",
)

INVALID_URLS = (
    # non-https schemes
    "http://www.youtube.com/watch?v=x",
    "ftp://youtube.com/x",
    "file:///etc/passwd",
    # local paths (previously treated as container-local files by the worker)
    "/data/video.mp4",
    "video.mp4",
    # SSRF targets
    "https://169.254.169.254/latest/meta-data/",
    "https://internal-service.local/video",
    "https://evil.com/watch?v=x",
    # host-suffix spoofing
    "https://youtube.com.evil.com/watch?v=x",
    "https://notyoutu.be/x",
)


@pytest.mark.parametrize("url", VALID_URLS)
def test_accepts_allowlisted_youtube_urls(url):
    request = SanjiJobRequest(input_url=url)
    assert request.input_url == url


@pytest.mark.parametrize("url", INVALID_URLS)
def test_rejects_disallowed_input_urls(url):
    with pytest.raises(ValidationError):
        SanjiJobRequest(input_url=url)


def test_worker_rejects_injected_local_path_message():
    """A message injected past the API must not reach the local-path fallback."""
    raw = json.dumps({"job_id": "j-1", "request_payload": {"input_url": "/etc/passwd"}})
    with pytest.raises(ValidationError):
        parse_job_message(raw)


def test_worker_accepts_valid_message():
    raw = json.dumps(
        {"job_id": "j-1", "request_payload": {"input_url": "https://youtu.be/x"}}
    )
    job_id, request = parse_job_message(raw)
    assert job_id == "j-1"
    assert request.input_url == "https://youtu.be/x"
