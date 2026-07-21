# sanji — dev deploy & teardown runbook

Single ordered procedure for standing up (and tearing down) the sanji SaaS service
tier in the **dev** account. The deploy knowledge previously lived scattered across
the README, each SAM template header, `build_and_push.sh`, and the sandjig README;
this file is the one place that sequences them. Addresses #77 and #11.

> **Ephemeral by policy.** Until the project is stable, dev is stood up only to test,
> capture evidence, then **torn down** (see [Teardown](#teardown)). Do not leave the
> stack running.

## Topology

Three planes, three tools, deployed in dependency order:

| # | Plane | Tool | Stack / artifact | Creates |
|---|-------|------|------------------|---------|
| 1 | jobs data-plane | sandjig CFN | `sandjig-resources-dev` | SQS `sanji-sdjobs-eph-dev` (+ DLQ), jobs/settings DynamoDB tables |
| 2 | SaaS data-plane | SAM/CFN | `sanji-resources-dev` | users/subscriptions/webhook/usage DynamoDB tables, results + uploads S3 buckets |
| 3 | worker image | Docker + ECR | `sanji-worker:<tag>` | container image the Batch job runs |
| 4 | compute-plane | SAM/CFN | `sanji-compute-dev` | Batch (Fargate Spot), EventBridge Pipe (SQS→Batch), Batch-FAILED rule, IAM |
| 5 | API | Zappa | Lambda `sanji-api-dev` | Flask API on Lambda + API Gateway; owns the results-bucket S3 notification |

Data flow proven by this topology (async):
`POST /uploads` → `POST /jobs` → SQS → EventBridge Pipe → Batch worker → S3 `result.json`
→ Zappa S3 event → `handle_result_event` → job terminal status → `GET /jobs/<id>/result`.

## Prerequisites

- `AWS_PROFILE=weyucou-dev-agent` (dev account `610714125210`, region `us-west-2`).
  Sanity-check before anything: `aws sts get-caller-identity --query Account --output text` must print `610714125210`.
- `uv`, Docker (with buildx for `linux/amd64`), the AWS CLI v2, and the `sandjig` CLI on PATH.
- A copy of `zappa_settings.json` created from the tracked example:
  `cp zappa_settings.example.json zappa_settings.json` and fill every `<PLACEHOLDER>`.
  The file is gitignored — never commit it.

### Survey the account first

Before deploying, confirm the target account is clean of prior stacks and pick the
network params the compute stack needs:

```bash
aws cloudformation list-stacks --query "StackSummaries[?starts_with(StackName, 'sanji') || starts_with(StackName, 'sandjig')].[StackName,StackStatus]" --output table
# Default-VPC public subnets + an egress security group for the Fargate worker:
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SUBNET_IDS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID Name=map-public-ip-on-launch,Values=true --query 'Subnets[].SubnetId' --output text | tr '\t' ',')
SG_ID=$(aws ec2 describe-security-groups --filters Name=vpc-id,Values=$VPC_ID Name=group-name,Values=default --query 'SecurityGroups[0].GroupId' --output text)
echo "SUBNET_IDS=$SUBNET_IDS  SG_ID=$SG_ID"
```

## Deploy

Run in order — each step depends on the previous one's resources.

### 1. sandjig jobs resources (`sandjig-resources-dev`)

Generates and deploys the data-plane-only template (SQS queue + jobs/settings tables).
The queue **must** resolve to `sanji-sdjobs-eph-dev` — the name is hard-coded in the
Zappa env (`PROCESSINGJOB_REQUEST_QUEUE_URL`) and the compute stack default
(`JobsQueueName`). This is the single source of truth for that name; changing it means
changing all three.

```bash
sandjig template --resources-only -o /tmp/sandjig-resources.sam.yaml   # run with sanji's suffix=eph / stage=dev config
aws cloudformation deploy \
  --template-file /tmp/sandjig-resources.sam.yaml \
  --stack-name sandjig-resources-dev \
  --capabilities CAPABILITY_IAM
```

See the [sandjig README → Deploy](https://github.com/monkut/sandjig#deploy) for the
exact `-s`/`-n` flags that produce the `sanji-sdjobs-eph-dev` name.

### 2. sanji SaaS resources (`sanji-resources-dev`)

```bash
aws cloudformation deploy \
  --template-file infra/sanji-resources.sam.yaml \
  --stack-name sanji-resources-dev \
  --parameter-overrides Stage=dev
```

Capture the bucket names the API + worker need:

```bash
aws cloudformation describe-stacks --stack-name sanji-resources-dev \
  --query "Stacks[0].Outputs[?OutputKey=='ResultsBucketName' || OutputKey=='UploadsBucketName'].[OutputKey,OutputValue]" --output text
```

### 3. Build & push the worker image

```bash
IMAGE_URI=$(AWS_PROFILE=weyucou-dev-agent infra/worker/build_and_push.sh dev)
echo "$IMAGE_URI"   # e.g. 610714125210.dkr.ecr.us-west-2.amazonaws.com/sanji-worker:dev
```

The script creates the `sanji-worker` ECR repo out-of-band (idempotent) and prints the
image URI on its last line — feed that into the next step.

### 4. sanji compute plane (`sanji-compute-dev`)

```bash
aws cloudformation deploy \
  --template-file infra/sanji-compute.sam.yaml \
  --stack-name sanji-compute-dev \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    Stage=dev \
    WorkerImageUri="$IMAGE_URI" \
    SubnetIds="$SUBNET_IDS" \
    SecurityGroupId="$SG_ID"
```

Imports the results/uploads bucket ARNs from `sanji-resources-dev` and the jobs queue by
name — steps 1 & 2 must exist first.

### 5. Deploy the API (Zappa)

```bash
uv sync --extra api --no-dev      # --no-dev is REQUIRED: the dev group (pyright/moto/hypothesis) busts Lambda's 250 MB unzipped limit (#77)
uv run zappa deploy dev           # first deploy;  use `zappa update dev` thereafter
```

Zappa owns the results-bucket `s3:ObjectCreated:*` (suffix `result.json`) →
`handle_result_event` notification via its `events` binding in `zappa_settings.json` —
that is why the resources stack does **not** declare it (avoids a CFN/Zappa conflict).

Before the first `POST /jobs`, confirm the Zappa env carries **every** table name,
including `SANJI_USERS_TABLE` and `SANJI_USAGE_TABLE`. If absent, the code falls back to
stage-less defaults (`sanji-users`, `sanji-usage`) that do not exist, and **every
authenticated request 401s** (#77). The tracked example now includes them.

### 6. Wire the Batch-FAILED safety net (manual)

Zappa 0.62 has no `event_pattern` support (it crashes with `UnboundLocalError: rule_name`),
so the Batch-FAILED → Lambda target is wired by hand after the API deploy. The compute
stack already created the `sanji-batch-failed-dev` rule; point it at the Lambda:

```bash
FN_ARN=$(aws lambda get-function --function-name sanji-api-dev --query 'Configuration.FunctionArn' --output text)
aws events put-targets --rule sanji-batch-failed-dev \
  --targets "Id=1,Arn=$FN_ARN,InputTransformer={InputPathsMap={detail=\$.detail},InputTemplate='{\"command\": \"sanji.service.results.handle_batch_state_change\", \"detail\": <detail>}'}"
aws lambda add-permission --function-name sanji-api-dev \
  --statement-id batch-failed-events --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "$(aws events describe-rule --name sanji-batch-failed-dev --query Arn --output text)"
```

## Verify (smoke test)

Run the deployed-environment smoke test against the API Gateway invoke URL that
`zappa deploy` printed:

```bash
API_URL=$(uv run zappa status dev --json | python -c "import sys,json;print(json.load(sys.stdin)['API Gateway URL'])")
python validation/smoke_test.py --api-url "$API_URL" --allowed-origin https://kanpaiko.weyuco.com
```

The script checks DNS/TLS, `GET /health` + sandjig `GET /healthcheck`, `GET /plans`,
credentialed CORS (allow-listed origin echoed with `Allow-Credentials: true`; disallowed
origin not echoed), and that `GET /me` / `POST /jobs` reject anonymous callers with 401
(not 500). It exits non-zero on any failure — usable as a post-deploy CI gate. See
[`validation/README.md`](../validation/README.md) for the manual (authenticated + full
async e2e) checklist that complements it.

## Teardown

Reverse order. Several steps are **not** handled by CloudFormation and will block a
naive `delete-stack`:

```bash
uv run zappa undeploy dev --remove-logs        # 1. FIRST — removes the results-bucket S3 notification; deleting the bucket before this conflicts
aws cloudformation delete-stack --stack-name sanji-compute-dev   # 2. compute before resources (it imports the resources exports)
aws cloudformation wait stack-delete-complete --stack-name sanji-compute-dev

# 3. Empty the S3 buckets — CloudFormation refuses to delete a non-empty bucket.
for B in $(aws cloudformation describe-stacks --stack-name sanji-resources-dev --query "Stacks[0].Outputs[?ends_with(OutputKey,'BucketName')].OutputValue" --output text); do
  aws s3 rm "s3://$B" --recursive
done
aws cloudformation delete-stack --stack-name sanji-resources-dev   # 4.
aws cloudformation delete-stack --stack-name sandjig-resources-dev # 5.

# 6. ECR repo is created out-of-band by build_and_push.sh — CFN never deletes it, and a non-empty repo blocks deletion:
aws ecr delete-repository --repository-name sanji-worker --force
```

### Teardown gotchas

- **`zappa undeploy` first** — it detaches the S3 → Lambda notification it created; deleting `sanji-resources-dev` while that notification points at a gone Lambda leaves the bucket in a bad state.
- **Empty buckets before delete** — the 7-day lifecycle rules mean results/uploads objects may still be present; a non-empty bucket blocks stack deletion. (Buckets are not versioned, so `s3 rm --recursive` suffices — no version/delete-marker purge needed.)
- **Compute before resources** — `sanji-compute-dev` `Fn::ImportValue`s the resources-stack exports; CloudFormation refuses to delete an exporting stack while an importer exists.
- **ECR is out-of-band** — the `sanji-worker` repo is created by the build script, not any stack; delete it manually with `--force`.
- **No `DeletionPolicy: Retain`** — deleting the resources stack **destroys** the DynamoDB tables and their data. That is intended for ephemeral dev, but be deliberate before running this against anything you care about.
