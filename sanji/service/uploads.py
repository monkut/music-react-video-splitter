"""Direct-upload presigned URL issuance (issue #65).

Users provide their reaction VOD by uploading it straight to S3 — the API only
signs URLs, the bytes never pass through Lambda. Small files get a single
presigned PUT (declared Content-Type/Content-Length are signed, so S3 rejects
a mismatch); larger files get a multipart part-URL set completed server-side
via ``handle_complete_upload``, which re-checks ownership and the size cap
(S3 cannot enforce total size across multipart parts).
"""

import math
import uuid
from typing import Any, TypeVar

import structlog
from botocore.exceptions import ClientError
from flask import Request, Response, jsonify
from pydantic import BaseModel, Field, ValidationError

from sanji.service.jobs import build_upload_key, is_user_upload_key
from sanji.settings import (
    UPLOAD_MAX_BYTES,
    UPLOAD_MULTIPART_THRESHOLD_BYTES,
    UPLOAD_PART_BYTES,
    UPLOAD_PRESIGN_EXPIRY_SECONDS,
    get_uploads_bucket,
)

logger = structlog.get_logger().bind(logger=__name__)

# S3 hard limit on multipart part count.
MAX_MULTIPART_PARTS = 10_000

# Accepted source video types → stored file extension.
CONTENT_TYPE_EXTENSIONS = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/quicktime": ".mov",
    "video/mpeg": ".mpg",
}


class UploadRequest(BaseModel):
    filename: str | None = None
    content_type: str
    content_length: int = Field(gt=0)


class UploadPart(BaseModel):
    part_number: int = Field(ge=1, le=MAX_MULTIPART_PARTS)
    etag: str


class CompleteUploadRequest(BaseModel):
    key: str
    s3_upload_id: str
    parts: list[UploadPart] = Field(min_length=1)


PydanticBaseModelType = TypeVar("PydanticBaseModelType", bound=BaseModel)


def _parse_body(
    request: Request, model: type[PydanticBaseModelType]
) -> PydanticBaseModelType | tuple[Response, int]:
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="invalid_request", message="JSON body required."), 422
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        return jsonify(error="invalid_request", message=str(exc)), 422


def handle_create_upload(
    request: Request, user_id: str, s3_client: Any
) -> tuple[Response, int]:
    """POST /uploads — issue presigned upload URL(s) under the user's prefix."""
    parsed = _parse_body(request, UploadRequest)
    if isinstance(parsed, tuple):
        return parsed
    upload = parsed

    # Enforce the video content-type allowlist: an unknown type is rejected
    # (415), an allowed one maps to the file extension the stored key gets.
    extension = CONTENT_TYPE_EXTENSIONS.get(upload.content_type)
    if extension is None:
        return jsonify(
            error="unsupported_content_type",
            message=f"content_type must be one of: "
            f"{', '.join(sorted(CONTENT_TYPE_EXTENSIONS))}",
        ), 415
    if upload.content_length > UPLOAD_MAX_BYTES:
        return jsonify(
            error="content_too_large",
            message=f"content_length exceeds the maximum of {UPLOAD_MAX_BYTES} bytes.",
        ), 413

    bucket = get_uploads_bucket()
    if not bucket:
        logger.error("uploads_bucket_unset")
        return jsonify(error="internal_server_error"), 500

    upload_id = uuid.uuid4().hex
    key = build_upload_key(user_id, upload_id, extension)
    # Response fields shared by both reply shapes (single PUT and multipart).
    common_response_fields = {
        "upload_id": upload_id,
        "key": key,
        "expires_in": UPLOAD_PRESIGN_EXPIRY_SECONDS,
    }

    if upload.content_length <= UPLOAD_MULTIPART_THRESHOLD_BYTES:
        # Signing ContentType + ContentLength makes S3 reject a PUT whose
        # headers differ from what was declared and validated here.
        url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ContentType": upload.content_type,
                "ContentLength": upload.content_length,
            },
            ExpiresIn=UPLOAD_PRESIGN_EXPIRY_SECONDS,
        )
        logger.info("upload_created", key=key, method="put", user_id=user_id)
        return jsonify(
            **common_response_fields,
            method="put",
            url=url,
            headers={
                "Content-Type": upload.content_type,
                "Content-Length": upload.content_length,
            },
        ), 201

    # Grow the part size when the default would exceed S3's 10,000-part limit.
    part_size = max(
        UPLOAD_PART_BYTES, math.ceil(upload.content_length / MAX_MULTIPART_PARTS)
    )
    part_count = math.ceil(upload.content_length / part_size)
    s3_upload_id = s3_client.create_multipart_upload(
        Bucket=bucket, Key=key, ContentType=upload.content_type
    )["UploadId"]
    part_urls = [
        {
            "part_number": part_number,
            "url": s3_client.generate_presigned_url(
                "upload_part",
                Params={
                    "Bucket": bucket,
                    "Key": key,
                    "UploadId": s3_upload_id,
                    "PartNumber": part_number,
                },
                ExpiresIn=UPLOAD_PRESIGN_EXPIRY_SECONDS,
            ),
        }
        for part_number in range(1, part_count + 1)
    ]
    logger.info(
        "upload_created", key=key, method="multipart", parts=part_count, user_id=user_id
    )
    return jsonify(
        **common_response_fields,
        method="multipart",
        s3_upload_id=s3_upload_id,
        part_size=part_size,
        part_urls=part_urls,
    ), 201


def handle_complete_upload(
    request: Request, user_id: str, s3_client: Any
) -> tuple[Response, int]:
    """POST /uploads/complete — finish a multipart upload and enforce the cap."""
    parsed = _parse_body(request, CompleteUploadRequest)
    if isinstance(parsed, tuple):
        return parsed
    complete = parsed

    if not is_user_upload_key(complete.key, user_id):
        logger.info(
            "upload_complete_ownership_denied", key=complete.key, user_id=user_id
        )
        return jsonify(
            error="forbidden",
            message="key is not under the authenticated user's uploads prefix.",
        ), 403

    bucket = get_uploads_bucket()
    if not bucket:
        logger.error("uploads_bucket_unset")
        return jsonify(error="internal_server_error"), 500

    try:
        s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=complete.key,
            UploadId=complete.s3_upload_id,
            MultipartUpload={
                "Parts": [
                    {"PartNumber": part.part_number, "ETag": part.etag}
                    for part in complete.parts
                ]
            },
        )
    except ClientError as exc:
        logger.info("upload_complete_failed", key=complete.key, error=str(exc))
        return jsonify(
            error="invalid_upload",
            message="multipart upload could not be completed.",
        ), 400

    size = s3_client.head_object(Bucket=bucket, Key=complete.key)["ContentLength"]
    if size > UPLOAD_MAX_BYTES:
        # The declared content_length was validated at create time, but nothing
        # stops a client uploading bigger parts — remove the oversize object.
        s3_client.delete_object(Bucket=bucket, Key=complete.key)
        logger.info("upload_oversize_deleted", key=complete.key, bytes=size)
        return jsonify(
            error="content_too_large",
            message=f"uploaded object exceeds the maximum of {UPLOAD_MAX_BYTES}"
            " bytes and was removed.",
        ), 413

    logger.info("upload_completed", key=complete.key, bytes=size, user_id=user_id)
    return jsonify(key=complete.key, size=size), 200
