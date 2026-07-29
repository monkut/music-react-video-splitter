"""One-shot backfill of the lifetime run counter (#91).

The free tier gained a 24-run lifetime ceiling stored in the usage table under
the sentinel period key ``TOTAL``. Users who signed up before that change have
monthly rows but no ``TOTAL`` row, so without this backfill every existing user
would silently start again from zero.

For each user the script sums their existing ``YYYY-MM`` rows and writes the
result to their ``TOTAL`` row with ``SET`` (not ``ADD``), which makes re-runs
idempotent — running it twice produces the same total rather than doubling it.

Usage:
    # report what would change, write nothing (default)
    python scripts/backfill_total_usage.py --table sanji-usage-dev

    # apply
    python scripts/backfill_total_usage.py --table sanji-usage-dev --apply

Run this as part of the deploy that ships the lifetime cap. Because the write
is a SET, an increment landing between the scan and the write can be
overwritten; run it before the new code serves traffic, or re-run it after.
"""

import argparse
import sys
from collections import defaultdict
from typing import Any

import boto3

from sanji.service.usage import DEFAULT_USAGE_TABLE, TOTAL_PERIOD_KEY


def compute_totals(table: Any) -> dict[str, int]:
    """Sum each user's monthly rows. The TOTAL sentinel row is excluded."""
    totals: dict[str, int] = defaultdict(int)
    scan_kwargs: dict[str, Any] = {}
    while True:
        response = table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            if item["period_key"] == TOTAL_PERIOD_KEY:
                continue
            totals[item["user_id"]] += int(item.get("stream_count", 0))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            return dict(totals)
        scan_kwargs["ExclusiveStartKey"] = last_key


def backfill_total_counts(
    table_name: str | None = None, *, apply: bool = False
) -> dict[str, int]:
    """Write each user's summed monthly count to their TOTAL row.

    Returns the computed per-user totals whether or not they were written, so
    a dry run reports exactly what an apply would do.
    """
    table = boto3.resource("dynamodb").Table(table_name or DEFAULT_USAGE_TABLE)
    totals = compute_totals(table)
    if apply:
        for user_id, total in totals.items():
            table.update_item(
                Key={"user_id": user_id, "period_key": TOTAL_PERIOD_KEY},
                UpdateExpression="SET stream_count = :total",
                ExpressionAttributeValues={":total": total},
            )
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=None, help="usage table name")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the TOTAL rows (default: report only)",
    )
    args = parser.parse_args(argv)

    totals = backfill_total_counts(args.table, apply=args.apply)
    mode = "applied" if args.apply else "dry-run"
    for user_id, total in sorted(totals.items()):
        print(f"{user_id}\t{total}")
    print(f"[{mode}] {len(totals)} user(s), {sum(totals.values())} run(s) total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
