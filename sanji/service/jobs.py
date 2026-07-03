"""sandjig job payload models for the sanji splitter (D15/#12, #13).

JobSpec in: what to split. JobResult out: where the results landed in S3.
``user_id`` rides inside the request payload until #6 (OAuth) lands —
see monkut/sandjig#3 for the upstream first-class field request.
"""

from urllib.parse import urlparse

from pydantic import Field, field_validator
from sandjig.jobsapi.validation.definitions import StatusSupportedValues
from sandjig.models import RequestPostPayloadBaseModel, ResponsePostPayloadBaseModel

# Exact-match host allowlist for job input URLs (#33). yt-dlp will fetch whatever
# URL it is handed from inside the Batch network, so only known YouTube hosts are
# accepted; anything else is an SSRF vector or a local-path escape.
ALLOWED_INPUT_URL_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)

# Terminal statuses the worker writes into result.json (mirror sandjig's vocabulary).
STATUS_COMPLETED = StatusSupportedValues.COMPLETED.value
STATUS_ERROR = StatusSupportedValues.ERROR.value


class SanjiJobRequest(RequestPostPayloadBaseModel):
    input_url: str = Field(
        ...,
        min_length=1,
        description="https YouTube URL (allowlisted hosts only)",
    )
    params: dict = Field(
        default_factory=dict, description="pipeline overrides (threshold, window, ...)"
    )
    user_id: str | None = Field(
        default=None, description="requesting user (set by the API, not the client)"
    )

    @field_validator("input_url")
    @classmethod
    def validate_input_url(cls, value: str) -> str:
        """Require an https URL on an allowlisted YouTube host (#33).

        Validated on the model so both entry points are covered: sandjig
        validates the API request body against this model (422 on failure),
        and the worker re-validates via ``parse_job_message`` — a message
        injected past the API (or a legacy job) is rejected before yt-dlp
        or any local-path fallback can touch it.
        """
        parsed = urlparse(value)
        if parsed.scheme != "https":
            raise ValueError("input_url must be an https URL")
        host = (parsed.hostname or "").lower()
        if host not in ALLOWED_INPUT_URL_HOSTS:
            raise ValueError(f"input_url host {host!r} is not an allowed YouTube host")
        return value


class SanjiJobResult(ResponsePostPayloadBaseModel):
    """Worker output. Written to S3 as ``result.json`` — the completion marker (#8).

    Doubles as the API ``response_payload`` (sandjig stores it on the Job record).
    ``job_id``/``status`` identify and classify the run; the remaining fields point
    at the artifacts the worker streamed to S3.
    """

    job_id: str | None = Field(
        default=None, description="owning job id (set by the worker in result.json)"
    )
    status: str = Field(
        default=STATUS_COMPLETED, description="terminal status: completed | error"
    )
    result_manifest_key: str | None = Field(
        default=None, description="S3 key of the manifest CSV"
    )
    segment_keys: list[str] = Field(
        default_factory=list, description="S3 keys of the split segments"
    )
    segment_count: int = Field(default=0, description="number of segments produced")
    duration: float | None = Field(
        default=None, description="source video duration in seconds"
    )
    error: str | None = Field(
        default=None, description="failure detail when status == error"
    )
