#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate multiple attention microbenchmark JSON results.

Supports two aggregation methods:
  - trimmed mean (default): drops N highest and N lowest values, averages the rest.
  - median: takes the median value across runs.

Outputs a new JSON in the same dashboard format, based on the first run's records.
"""

import argparse
import json
import math
import statistics


def make_key(record):
    """Unique key for a benchmark case."""
    model_name = record["model"]["name"]
    metric_name = record["metric"]["name"]
    shape = record["benchmark"]["extra_info"].get("shape", "")
    return f"{model_name}|{metric_name}|{shape}"


def trimmed_mean(values, trim=2):
    """Drop `trim` max and `trim` min values, average the rest.

    Falls back to a plain mean if there are not enough values to trim.
    """
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan")
    clean.sort()
    if len(clean) > 2 * trim:
        clean = clean[trim:-trim]
    return statistics.mean(clean)


def aggregate(values, method, trim):
    """Dispatch to the chosen aggregation method."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan")
    if method == "median":
        return statistics.median(clean)
    return trimmed_mean(clean, trim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input JSON result files")
    parser.add_argument("--output", required=True, help="Output aggregated JSON file")
    parser.add_argument(
        "--trim", type=int, default=2, help="Number of max/min values to drop (each, trimmed mean only)"
    )
    parser.add_argument(
        "--method", choices=["trimmed_mean", "median"], default="trimmed_mean",
        help="Aggregation method (default: trimmed_mean)"
    )
    args = parser.parse_args()

    # Collect values per case across all runs
    all_values = {}  # key -> list of values
    base_records = None  # records from the first run, used as output template
    for i, path in enumerate(args.inputs):
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        if i == 0:
            base_records = records
        for record in records:
            key = make_key(record)
            values = record["metric"]["benchmark_values"]
            if values:
                all_values.setdefault(key, []).append(values[0])

    # Build aggregated output from the first run's records
    aggregated = []
    for record in base_records:
        key = make_key(record)
        values = all_values.get(key, [])
        agg_value = aggregate(values, args.method, args.trim)
        record["metric"]["benchmark_values"] = [agg_value]
        if args.method == "median":
            agg_desc = "median"
        else:
            agg_desc = f"trimmed mean (drop {args.trim} max + {args.trim} min)"
        record["metric"]["extra_info"] = {
            "aggregation": agg_desc,
            "num_runs": len(values),
        }
        aggregated.append(record)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Aggregated {len(args.inputs)} runs, {len(aggregated)} cases -> {args.output}")


if __name__ == "__main__":
    main()
