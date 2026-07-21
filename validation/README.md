# sanji deployment validation

Two tiers of post-deploy verification for a live sanji deployment.

## Tier 1 — automated smoke test

`smoke_test.py` hits a live API Gateway URL and asserts the machine-checkable
contract (DNS, TLS, public endpoints, CORS, auth boundary). Stdlib-only, exits
non-zero on any failure, so it can gate a deploy in CI.

```bash
python validation/smoke_test.py \
  --api-url "$API_URL" \
  --allowed-origin https://kanpaiko.weyuco.com
```

`$API_URL` is the invoke URL `zappa deploy dev` prints (includes the `/dev`
stage path). See [`infra/DEPLOY.md`](../infra/DEPLOY.md) for how it fits into the
deploy sequence.

## Tier 2 — manual checklist

The smoke test deliberately stops at the auth boundary. These require a real
Google account, session cookie, and video, so they are verified by hand and the
results recorded in the deploy tracking issue (#11).

- [ ] `GET /auth/google` redirects to Google; completing consent returns to the
      app authenticated with a session cookie.
- [ ] `GET /me` (authenticated) returns the current user.
- [ ] `GET /me/usage` returns the current-period stream count, plan limit, and
      subscription status.
- [ ] `GET /plans` lists the Free / Pro / Business plans with correct limits.
- [ ] **Async e2e:** `POST /uploads` returns a presigned PUT → PUT a source VOD →
      `POST /jobs` with the `source_s3_key` returns 201 → poll `GET /jobs/<id>`
      until terminal → `GET /jobs/<id>/result` returns presigned segment +
      manifest URLs that download.
- [ ] A second user's `GET /jobs/<id>/result` for a job they do not own returns
      404 (ownership isolation, #30).
- [ ] Over-quota `POST /jobs` returns 402 once the plan's monthly stream cap is hit.
- [ ] `POST /billing/checkout` (authenticated) returns a Stripe Checkout URL.
- [ ] Batch-FAILED path: force a worker failure and confirm the job reaches a
      terminal failed status via `handle_batch_state_change` (validates the
      manual EventBridge wiring from DEPLOY.md step 6).
- [ ] AWS console: Batch job ran on Fargate Spot; results bucket holds
      `result.json`; no unexpected CloudWatch errors in the Lambda log group.
