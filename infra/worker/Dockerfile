# syntax=docker/dockerfile:1
#
# AWS Batch worker image for the sanji pipeline (#8 / #11).
#
# Runs run_pipeline() over a JobSpec (delivered as JSON via the SANJI_JOB_MESSAGE
# env var by the Batch job / EventBridge Pipe), then lands the segments, manifest,
# and result.json in S3. The worker touches S3 only (s3:PutObject).
#
# Platform: linux/amd64. D2 = x86 for the first deploy; ARM/Graviton is the
# Phase-2 cost lever and needs an arm64 build check of ctranslate2 + ffmpeg.
#
# Build context is the repo root:
#   docker build -f infra/worker/Dockerfile -t sanji-worker .

FROM --platform=linux/amd64 python:3.14-slim-bookworm

# ffmpeg: yt-dlp download + segment splitting. git: sandjig is a git-pinned dep.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# uv for dependency install (pinned to the project toolchain version).
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install only the `worker` extra (no Flask/Zappa/api deps). --frozen pins to uv.lock.
COPY pyproject.toml uv.lock README.md ./
COPY sanji ./sanji
RUN uv sync --no-dev --extra worker --frozen

ENV PATH="/app/.venv/bin:${PATH}"

# NOTE: faster-whisper downloads its model from HuggingFace on first run (the
# worker has outbound internet via the public subnet). Bake the model into the
# image in a later iteration if cold-start latency matters.

# Batch delivers the JobSpec JSON via SANJI_JOB_MESSAGE.
ENTRYPOINT ["python", "-m", "sanji.worker"]
