"""sanji deployed-environment smoke test.

Runs the automatable checks against a live sanji API Gateway deployment
(the invoke URL `zappa deploy dev` prints, including the stage path, e.g.
`https://abc123.execute-api.us-west-2.amazonaws.com/dev`).

Usage:
    python validation/smoke_test.py \\
        --api-url https://abc123.execute-api.us-west-2.amazonaws.com/dev \\
        --allowed-origin https://kanpaiko.weyuco.com

Exits 0 only when every automated check passes, so it can gate a deploy in
CI. The authenticated happy-path and the full async e2e (upload -> jobs ->
Batch -> result) require a session cookie / real video and are covered by the
manual checklist in validation/README.md.

Automated checks:
    Infrastructure:
        1 API hostname resolves via DNS
        2 TLS certificate is valid, trusted, and issued for the host
    Public API:
        3 GET /health returns 200 with {"status": "ok"}
        4 GET /healthcheck (sandjig) returns 200
        5 GET /plans returns 200 and a non-empty plans list
    CORS:
        6 Preflight from the allow-listed origin echoes that origin with
          Access-Control-Allow-Credentials: true (never a wildcard)
        7 Preflight from a disallowed origin does NOT echo it
    Auth boundary:
        8 GET /me while anonymous returns 401 (not 500)
        9 POST /jobs while anonymous returns 401 (not 500)
"""

# ruff: noqa: T201, S113

from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener

REQUEST_TIMEOUT_SECONDS = 15
TLS_EXPIRY_WARNING_DAYS = 30
DEFAULT_DISALLOWED_ORIGIN = "https://evil.example.com"


def _opener():
    return build_opener()


def _record(results: list, label: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    results.append((label, passed, detail))


def _get(url: str, method: str = "GET", headers: dict | None = None):
    """Return (status, headers, body) surfacing HTTPError as a normal response."""
    req = Request(url, method=method, headers=headers or {})
    try:
        resp = _opener().open(req, timeout=REQUEST_TIMEOUT_SECONDS)
        return resp.status, resp.headers, resp.read()
    except HTTPError as exc:
        return exc.code, exc.headers, exc.read() if exc.fp else b""
    except URLError as exc:
        # connection refused / DNS failure / timeout — surface as status 0 so the
        # calling check records a FAIL instead of crashing the whole run.
        print(f"    (request error for {url}: {exc.reason})")
        return 0, None, b""


def check_dns(results: list, url: str) -> None:
    host = urlparse(url).hostname
    if not host:
        _record(results, "API hostname resolves", False, f"no hostname in {url}")
        return
    try:
        addrs = socket.getaddrinfo(host, None)
        addr = addrs[0][4][0] if addrs else "<none>"
        _record(results, "API hostname resolves", True, f"{host} -> {addr}")
    except socket.gaierror as exc:
        _record(results, "API hostname resolves", False, f"{host}: {exc}")


def check_tls(results: list, url: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 443
    if not host:
        _record(results, "TLS certificate valid for host", False, f"no hostname in {url}")
        return
    context = ssl.create_default_context()
    try:
        with socket.create_connection((host, port), timeout=REQUEST_TIMEOUT_SECONDS) as raw:
            with context.wrap_socket(raw, server_hostname=host) as tls:
                cert = tls.getpeercert()
    except (ssl.SSLError, socket.gaierror, socket.timeout, OSError) as exc:
        _record(results, "TLS certificate valid for host", False, f"{host}: {exc}")
        return
    not_after = cert.get("notAfter") if cert else None
    expires_at = (
        datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        if not_after
        else None
    )
    days_left = (expires_at - datetime.now(tz=timezone.utc)).days if expires_at else -1
    passed = days_left > TLS_EXPIRY_WARNING_DAYS
    _record(results, "TLS certificate valid for host", passed, f"expires_in_days={days_left}")


def check_health(results: list, base: str) -> None:
    status, _, body = _get(base + "/health")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        payload = {}
    ok = status == 200 and payload.get("status") == "ok"
    _record(results, "GET /health returns 200 {status: ok}", ok, f"status={status}, body={payload}")


def check_healthcheck(results: list, base: str) -> None:
    status, _, _ = _get(base + "/healthcheck")
    _record(results, "GET /healthcheck (sandjig) returns 200", status == 200, f"status={status}")


def check_plans(results: list, base: str) -> None:
    status, _, body = _get(base + "/plans")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        payload = {}
    plans = payload.get("plans") if isinstance(payload, dict) else None
    ok = status == 200 and bool(plans)
    detail = f"status={status}, plans={len(plans) if plans else 0}"
    _record(results, "GET /plans returns 200 with plans", ok, detail)


def _preflight_allow_origin(base: str, origin: str) -> str:
    _, headers, _ = _get(
        base + "/me",
        method="OPTIONS",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    return headers.get("Access-Control-Allow-Origin", "") if headers else ""


def check_cors_allows_origin(results: list, base: str, origin: str) -> None:
    _, headers, _ = _get(
        base + "/me",
        method="OPTIONS",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    allow_origin = headers.get("Access-Control-Allow-Origin", "") if headers else ""
    allow_creds = headers.get("Access-Control-Allow-Credentials", "") if headers else ""
    echoed = allow_origin.rstrip("/") == origin.rstrip("/")
    passed = echoed and allow_origin != "*" and allow_creds.lower() == "true"
    detail = f"Allow-Origin={allow_origin or '<missing>'}, Allow-Credentials={allow_creds or '<missing>'}"
    _record(results, "CORS echoes allow-listed origin + credentials", passed, detail)


def check_cors_rejects_origin(results: list, base: str, disallowed: str) -> None:
    allow_origin = _preflight_allow_origin(base, disallowed)
    rejected = allow_origin not in (disallowed, "*") or not allow_origin
    detail = f"Allow-Origin={allow_origin or '<missing>'} for Origin={disallowed}"
    _record(results, "CORS rejects disallowed origin", rejected, detail)


def check_auth_required(results: list, base: str, method: str, path: str) -> None:
    status, _, _ = _get(base + path, method=method)
    passed = status == 401
    _record(results, f"{method} {path} anonymous -> 401 (not 500)", passed, f"status={status}")


def main() -> int:
    parser = argparse.ArgumentParser(description="sanji deployed-environment smoke test")
    parser.add_argument(
        "--api-url",
        required=True,
        help="Deployed API base URL incl. stage path (https://...execute-api.../dev)",
    )
    parser.add_argument(
        "--allowed-origin",
        default="https://kanpaiko.weyuco.com",
        help="Origin that must be allowed by CORS (SANJI_CORS_ALLOWED_ORIGINS)",
    )
    parser.add_argument(
        "--disallowed-origin",
        default=DEFAULT_DISALLOWED_ORIGIN,
        help="Origin that must be rejected by CORS",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON results after the report")
    args = parser.parse_args()

    base = args.api_url.rstrip("/")
    print(f"sanji smoke test — api={base}")
    print("=" * 70)

    results: list[tuple[str, bool, str]] = []

    print("\n--- Infrastructure ---")
    check_dns(results, base)
    check_tls(results, base)

    print("\n--- Public API ---")
    check_health(results, base)
    check_healthcheck(results, base)
    check_plans(results, base)

    print("\n--- CORS ---")
    check_cors_allows_origin(results, base, args.allowed_origin)
    check_cors_rejects_origin(results, base, args.disallowed_origin)

    print("\n--- Auth boundary ---")
    check_auth_required(results, base, "GET", "/me")
    check_auth_required(results, base, "POST", "/jobs")

    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    print("\n" + "=" * 70)
    print(f"Automated results: {passed} passed, {failed} failed, {len(results)} total")
    print("\nManual checks (record in the deploy issue) — see validation/README.md:")
    print("  - Google OAuth sign-in completes and sets a session cookie")
    print("  - Authenticated GET /me and GET /me/usage return the user + quota")
    print("  - Full async e2e: POST /uploads -> PUT source -> POST /jobs ->")
    print("    poll GET /jobs/<id>/result until segments + manifest URLs return")

    if args.json:
        print()
        print(
            json.dumps(
                {
                    "api_url": base,
                    "results": [
                        {"label": label, "passed": ok, "detail": detail}
                        for label, ok, detail in results
                    ],
                    "passed": passed,
                    "failed": failed,
                }
            )
        )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
