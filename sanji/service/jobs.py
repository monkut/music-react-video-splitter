"""sandjig job payload models for the sanji splitter (D15/#12, #13).

JobSpec in: what to split. JobResult out: where the results landed in S3.
``user_id`` rides inside the request payload until #6 (OAuth) lands —
see monkut/sandjig#3 for the upstream first-class field request.
"""

from pydantic import Field
from sandjig.models import RequestPostPayloadBaseModel, ResponsePostPayloadBaseModel


class SanjiJobRequest(RequestPostPayloadBaseModel):
    input_url: str = Field(
        ..., min_length=1, description="YouTube URL or source video location"
    )
    params: dict = Field(
        default_factory=dict, description="pipeline overrides (threshold, window, ...)"
    )
    user_id: str | None = Field(
        default=None, description="requesting user (set by the API, not the client)"
    )


class SanjiJobResult(ResponsePostPayloadBaseModel):
    result_manifest_key: str | None = Field(
        default=None, description="S3 key of the manifest CSV"
    )
    segment_keys: list[str] = Field(
        default_factory=list, description="S3 keys of the split segments"
    )
    segment_count: int = Field(default=0, description="number of segments produced")
