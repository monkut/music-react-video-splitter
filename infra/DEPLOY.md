# sanji — dev deploy & teardown runbook

Single ordered procedure for standing up (and tearing down) the sanji SaaS service
tier in the **dev** account. The deploy knowledge previously lived scattered across
the README, each SAM template header, `build_and_push.sh`, and the sandjig README;
this file is the one place that sequences them. Addresses #77 and #11.

> **Ephemeral by policy.** Until the project is stable, dev is stood up only to test,
> capture evidence, then **torn down** (see [Teardown](#teardown)). Do not leave the
> stack running.

> **Validated end-to-end on 2026-07-21** against dev `610714125210`/us-west-2: all five
> deploy steps, the manual Batch-FAILED wiring, the smoke test (9/9), and the full
> teardown were executed from this document. Commands below are the ones that actually ran.

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

The template is a static file in the sandjig checkout (`sandjig template --resources-only`
just copies it), so deploy it directly. The queue name is built as
`${Prefix}-sdjobs-${UniqueSuffix}-${APIGatewayStage}` — the parameters below are what
produce `sanji-sdjobs-eph-dev`:

```bash
aws cloudformation deploy \
  --template-file ~/projects/sandjig/sandjig/cloudformation/resources.sam.yaml \
  --stack-name sandjig-resources-dev \
  --region us-west-2 \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides Prefix=sanji UniqueSuffix=eph APIGatewayStage=dev \
    DynamodbRequestsTableName=sanji-jobs-requests-dev \
    DynamodbSettingsTableName=sanji-jobs-settings-dev \
    DyanmodbSortIndexName=sanji-jobs-sortindex-dev
```

> `DyanmodbSortIndexName` is spelled exactly that way in the template (typo upstream) —
> passing `Dynamodb...` fails with "invalid parameter key".

Verify the queue name matches the Zappa env before continuing:

```bash
aws sqs get-queue-url --queue-name sanji-sdjobs-eph-dev --region us-west-2 --query QueueUrl --output text
# must equal .dev.environment_variables.PROCESSINGJOB_REQUEST_QUEUE_URL in zappa_settings.json
```

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

### Google OAuth — required before first login

The API's `/auth/google` login flow requires three env vars in `zappa_settings.json`. Without them, the backend redirects to Google with no `client_id` and Google returns **Error 400: missing required parameter: client_id**.

**Step 1 — Create a Google Cloud OAuth 2.0 Client**

1. Open [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials
2. Click **Create Credentials → OAuth 2.0 Client ID**
3. Application type: **Web application**
4. Under **Authorized JavaScript origins**, add:
   - `https://kanpaiko.weyuco.com`
   - `https://dev.kanpaiko.weyuco.com`
   - `http://localhost:5173`
5. Under **Authorized redirect URIs**, add:
   - `http://localhost:5000/auth/google/callback` (local dev)
   - `https://dev.kanpaiko.weyuco.com/auth/google/callback/` (dev deploy — custom domain for the sanji API)
6. Click **Create**. Copy the **Client ID** (ends in `.apps.googleusercontent.com`) and **Client Secret**.

> If the Google Cloud project has not yet had the **Google People API** (or **Google+ API**) enabled, enable it under APIs & Services → Library.

**Custom domain for the dev API**: `dev.kanpaiko.weyuco.com` is a Route 53 ALIAS pointing to the API Gateway EDGE CloudFront distribution. It routes all requests through to the `dev` stage of the sanji API. The API Gateway base path mapping is a root mapping (no prefix) so all paths are forwarded verbatim — `/auth/google/callback/` at the domain maps to `/auth/google/callback/` in the Flask app.

**Step 2 — Add env vars to `zappa_settings.json`**

```json
"GOOGLE_CLIENT_ID": "<copied client id>",
"GOOGLE_CLIENT_SECRET": "<copied client secret>",
"OAUTH_REDIRECT_URI": "https://dev.kanpaiko.weyuco.com/auth/google/callback/"
```

Then redeploy: `uv run zappa update dev`

**Important:** `OAUTH_REDIRECT_URI` must exactly match one of the URIs registered in Google Cloud Console — even a trailing slash difference causes a redirect_uri_mismatch error.

**Step 3 — For the deployed frontend at `kanpaiko.weyuco.com`**

`https://kanpaiko.weyuco.com` and `https://dev.kanpaiko.weyuco.com` must be listed as **Authorized JavaScript origins** in the Google Cloud Console OAuth client (see step 1 above). This allows the SPA to initiate the OAuth flow from the custom domains.

### `SANJI_CORS_ALLOWED_ORIGINS` is stage-dependent

`get_cors_allowed_origins()` (`sanji/settings.py`) defaults to an **empty allowlist** when
the variable is unset — the API then rejects *every* browser origin, and the SPA cannot
call it at all. It must be set explicitly per stage:

| Stage | Value |
|-------|-------|
| dev (local SPA against a deployed API) | `http://localhost:5173` |
| prod | `https://kanpaiko.weyuco.com` (the #7 CloudFront domain) |

Comma-separate to allow more than one. The smoke test's CORS checks verify both that the
allow-listed origin is echoed with `Access-Control-Allow-Credentials: true` and that an
unknown origin is not.

> Packaging note: `uv sync --extra api --no-dev` keeps the upload around **44 MB** —
> well inside Lambda's 250 MB unzipped limit (verified on the 2026-07-21 dev deploy).

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
# 1. FIRST — detaches the results-bucket S3 notification (and the batch-failed target);
#    deleting the bucket/stack before this conflicts. -y skips the confirmation prompt.
uv run zappa undeploy dev --remove-logs -y

# 2. Compute before resources (it Fn::ImportValues the resources exports).
aws cloudformation delete-stack --stack-name sanji-compute-dev --region us-west-2
aws cloudformation wait stack-delete-complete --stack-name sanji-compute-dev --region us-west-2

# 3. Empty the S3 buckets — CloudFormation refuses to delete a non-empty bucket.
for B in sanji-results-dev-$(aws sts get-caller-identity --query Account --output text) \
         sanji-uploads-dev-$(aws sts get-caller-identity --query Account --output text); do
  aws s3 rm "s3://$B" --recursive
done

# 4/5. Resources stacks.
aws cloudformation delete-stack --stack-name sanji-resources-dev --region us-west-2
aws cloudformation wait stack-delete-complete --stack-name sanji-resources-dev --region us-west-2
aws cloudformation delete-stack --stack-name sandjig-resources-dev --region us-west-2
aws cloudformation wait stack-delete-complete --stack-name sandjig-resources-dev --region us-west-2

# 6. ECR repo is created out-of-band by build_and_push.sh — CFN never deletes it, and a non-empty repo blocks deletion:
aws ecr delete-repository --repository-name sanji-worker --region us-west-2 --force

# 7. Zappa's deployment-artifact bucket (`s3_bucket` in zappa_settings.json). Zappa
#    CREATES this on first deploy but never removes it on undeploy — it survives an
#    otherwise-complete teardown and is easy to miss.
aws s3 rm "s3://sanji-api-zappa-<acct>-usw2" --recursive
aws s3api delete-bucket --bucket "sanji-api-zappa-<acct>-usw2" --region us-west-2
```

Confirm nothing survived:

```bash
aws cloudformation list-stacks --region us-west-2 --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE DELETE_FAILED \
  --query "StackSummaries[?starts_with(StackName,'sanji')||starts_with(StackName,'sandjig')].[StackName,StackStatus]" --output text
aws ecr describe-repositories --region us-west-2 --query 'repositories[].repositoryName' --output text
aws s3api list-buckets --query "Buckets[?starts_with(Name,'sanji')].Name" --output text
aws dynamodb list-tables --region us-west-2 --query "TableNames[?starts_with(@,'sanji')]" --output text
aws batch describe-job-queues --region us-west-2 --query 'jobQueues[].jobQueueName' --output text
aws pipes list-pipes --region us-west-2 --query 'Pipes[].Name' --output text
```

### Teardown gotchas

- **`zappa undeploy` first** — it detaches the S3 → Lambda notification it created; deleting `sanji-resources-dev` while that notification points at a gone Lambda leaves the bucket in a bad state.
- **Empty buckets before delete** — the 7-day lifecycle rules mean results/uploads objects may still be present; a non-empty bucket blocks stack deletion. (Buckets are not versioned, so `s3 rm --recursive` suffices — no version/delete-marker purge needed.)
- **Compute before resources** — `sanji-compute-dev` `Fn::ImportValue`s the resources-stack exports; CloudFormation refuses to delete an exporting stack while an importer exists.
- **ECR is out-of-band** — the `sanji-worker` repo is created by the build script, not any stack; delete it manually with `--force`.
- **No `DeletionPolicy: Retain`** — deleting the resources stack **destroys** the DynamoDB tables and their data. That is intended for ephemeral dev, but be deliberate before running this against anything you care about.
- **Zappa's deploy bucket outlives the teardown** — `zappa undeploy` removes the Lambda, log group, S3 notification and event targets, but **not** the `s3_bucket` it created for deployment artifacts. Step 7 above exists because a full stack teardown otherwise leaves that bucket behind (found during the 2026-07-21 validation run).
