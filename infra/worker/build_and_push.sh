#!/usr/bin/env bash
#
# Build and push the sanji Batch worker image to ECR (dev account, us-west-2).
#
# Auth is taken from the ambient shell (set AWS_PROFILE before running), e.g.:
#   AWS_PROFILE=weyucou-dev-agent infra/worker/build_and_push.sh [tag]
#
# Prints the pushed image URI on the last line (consumed by the deploy runbook).
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-2}"
ECR_REPO="${ECR_REPO:-sanji-worker}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_TAG="${1:-$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo ">> account=${ACCOUNT_ID} region=${AWS_REGION} image=${IMAGE_URI}" >&2

# Ensure the ECR repo exists (idempotent).
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true >/dev/null

# Authenticate Docker to ECR.
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY" >&2

# Build (x86_64) and push.
docker build --platform linux/amd64 -f "${REPO_ROOT}/infra/worker/Dockerfile" -t "$IMAGE_URI" "$REPO_ROOT" >&2
docker push "$IMAGE_URI" >&2

echo ">> pushed ${IMAGE_URI}" >&2
echo "$IMAGE_URI"
