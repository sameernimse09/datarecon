from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .core import reconcile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datarecon", description="Explainable dataset reconciliation.")
    sub = parser.add_subparsers(dest="command", required=True)

    cmp = sub.add_parser("compare", help="Compare two files.")
    cmp.add_argument("source")
    cmp.add_argument("target")
    cmp.add_argument("--key", action="append", required=True, help="Key column; repeat or comma-separate.")
    cmp.add_argument("--output", "-o", help="Write report to .xlsx (full) or .csv (differences only).")
    cmp.add_argument("--tolerance", action="append", default=[], metavar="COL=VALUE")
    cmp.add_argument("--fuzzy", type=float, help="Similarity threshold (0-1) for near-identical text.")
    cmp.add_argument("--ignore-column", action="append", default=[])
    cmp.add_argument("--treat-as-equal", default="", help="Comma-separated causes, e.g. case,whitespace.")
    cmp.add_argument("--fail-on-diff", action="store_true", help="Exit with status 1 if differences exist.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    keys = [k.strip() for entry in args.key for k in entry.split(",") if k.strip()]
    rules = {}
    for item in args.tolerance:
        column, _, value = item.partition("=")
        rules[column] = {"tolerance": float(value)}

    result = reconcile(
        args.source,
        args.target,
        keys,
        rules=rules,
        ignore_columns=args.ignore_column,
        treat_as_equal={c.strip() for c in args.treat_as_equal.split(",") if c.strip()},
        fuzzy_threshold=args.fuzzy,
    )
    sys.stdout.write(result.summary() + "\n")

    if args.output:
        if args.output.lower().endswith(".csv"):
            result.differences.to_csv(args.output, index=False)
        else:
            result.to_excel(args.output)

    return 1 if (args.fail_on_diff and result.has_differences) else 0


if __name__ == "__main__":
    raise SystemExit(main())
