#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare aggregated attention microbenchmark results across rounds.

Takes two or more aggregated JSON result files (one per round, produced
by aggregate_results.py) and reports the per-case difference between each
round and the first one (baseline). Prints a human-readable summary
(overall + per-backend) and writes a detailed JSON diff report.
"""

import argparse
import json
import math
import sys


def make_key(record):
    """Unique key for a benchmark case."""
    model_name = record["model"]["name"]
    metric_name = record["metric"]["name"]
    shape = record["benchmark"]["extra_info"].get("shape", "")
    return f"{model_name}|{metric_name}|{shape}"


def get_backend(record):
    """Extract the backend name from the model name (first '_' token)."""
    return record["model"]["name"].split("_", 1)[0]


def get_value(record):
    values = record["metric"]["benchmark_values"]
    return values[0] if values else None


def load_records(path):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return {make_key(r): r for r in records}


def summarize(diffs):
    """Compute summary stats from a list of relative diffs (fractions)."""
    clean = [d for d in diffs if d is not None and not math.isnan(d)]
    if not clean:
        return {"count": 0}
    abs_clean = [abs(d) for d in clean]
    return {
        "count": len(clean),
        "mean_rel_diff": sum(clean) / len(clean),
        "mean_abs_rel_diff": sum(abs_clean) / len(abs_clean),
        "max_abs_rel_diff": max(abs_clean),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Aggregated JSON files (one per round)")
    parser.add_argument("--output", required=True, help="Output diff JSON file")
    parser.add_argument(
        "--metric-contains",
        default=None,
        help="Only compare metrics whose name contains this substring (e.g. 'latency')",
    )
    parser.add_argument(
        "--fail-above",
        type=float,
        default=None,
        help="Exit with code 1 if the max |rel diff| exceeds this fraction (e.g. 0.05 for 5%%)",
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("Need at least 2 input files to compare")

    baseline = load_records(args.inputs[0])
    base_name = args.inputs[0]

    all_diffs = []          # per-case detailed diffs (vs baseline)
    overall_rel = []        # all relative diffs (vs baseline)
    per_backend_rel = {}    # backend -> list of relative diffs

    for path in args.inputs[1:]:
        other = load_records(path)
        for key, base_rec in baseline.items():
            if key not in other:
                continue
            metric_name = base_rec["metric"]["name"]
            if args.metric_contains and args.metric_contains not in metric_name:
                continue
            backend = get_backend(base_rec)
            base_val = get_value(base_rec)
            other_val = get_value(other[key])
            if base_val is None or other_val is None:
                continue
            if math.isnan(base_val) or math.isnan(other_val):
                continue

            abs_diff = other_val - base_val
            rel_diff = abs_diff / base_val if base_val != 0 else float("nan")

            all_diffs.append(
                {
                    "case": key,
                    "backend": backend,
                    "metric": base_rec["metric"]["name"],
                    "baseline": base_val,
                    "compared": other_val,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "compared_file": path,
                }
            )
            overall_rel.append(rel_diff)
            per_backend_rel.setdefault(backend, []).append(rel_diff)

    # Sort detailed diffs by absolute relative diff (largest first)
    all_diffs.sort(key=lambda d: abs(d["rel_diff"]) if not math.isnan(d["rel_diff"]) else -1, reverse=True)

    overall_summary = summarize(overall_rel)
    backend_summary = {b: summarize(v) for b, v in sorted(per_backend_rel.items())}

    # Print human-readable summary
    print(f"Baseline: {base_name}")
    print(f"Compared rounds: {len(args.inputs) - 1}, matched cases: {len(all_diffs)}")
    print("\n=== Overall (relative diff vs baseline) ===")
    if overall_summary.get("count"):
        print(f"  cases:            {overall_summary['count']}")
        print(f"  mean rel diff:    {overall_summary['mean_rel_diff'] * 100:+.3f}%")
        print(f"  mean |rel diff|:  {overall_summary['mean_abs_rel_diff'] * 100:.3f}%")
        print(f"  max  |rel diff|:  {overall_summary['max_abs_rel_diff'] * 100:.3f}%")

    print("\n=== Per backend (mean |rel diff| / max |rel diff|) ===")
    for b, s in backend_summary.items():
        if s.get("count"):
            print(
                f"  {b:<12} n={s['count']:<6} "
                f"mean|d|={s['mean_abs_rel_diff'] * 100:7.3f}%  "
                f"max|d|={s['max_abs_rel_diff'] * 100:7.3f}%"
            )

    print("\n=== Top 10 cases by |rel diff| ===")
    for d in all_diffs[:10]:
        print(
            f"  {d['backend']:<10} {d['metric']:<18} "
            f"{d['baseline']:10.3f} -> {d['compared']:10.3f}  "
            f"({d['rel_diff'] * 100:+.2f}%)  {d['case']}"
        )

    report = {
        "baseline": base_name,
        "compared_files": args.inputs[1:],
        "num_cases": len(all_diffs),
        "overall": overall_summary,
        "per_backend": backend_summary,
        "diffs": all_diffs,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed diff report saved to {args.output}")

    # Optional pass/fail gate based on the max absolute relative diff.
    if args.fail_above is not None:
        if not overall_summary.get("count"):
            print("\nRESULT: FAIL - no comparable cases found")
            sys.exit(1)
        max_abs = overall_summary["max_abs_rel_diff"]
        threshold_pct = args.fail_above * 100
        if max_abs > args.fail_above:
            print(
                f"\nRESULT: FAIL - max |rel diff| {max_abs * 100:.3f}% "
                f"exceeds threshold {threshold_pct:.3f}%"
            )
            # List every case that exceeds the threshold (all_diffs is already
            # sorted by |rel diff| descending, so worst offenders come first).
            violations = [
                d
                for d in all_diffs
                if not math.isnan(d["rel_diff"]) and abs(d["rel_diff"]) > args.fail_above
            ]
            print(
                f"\n=== Cases exceeding {threshold_pct:.3f}% threshold "
                f"({len(violations)} cases) ==="
            )
            for d in violations:
                print(
                    f"  {d['backend']:<10} {d['metric']:<18} "
                    f"{d['baseline']:10.3f} -> {d['compared']:10.3f}  "
                    f"({d['rel_diff'] * 100:+.2f}%)  {d['case']}"
                )
            sys.exit(1)
        print(
            f"\nRESULT: PASS - max |rel diff| {max_abs * 100:.3f}% "
            f"within threshold {threshold_pct:.3f}%"
        )


if __name__ == "__main__":
    main()
