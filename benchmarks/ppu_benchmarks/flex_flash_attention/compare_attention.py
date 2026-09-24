#!/usr/bin/env python3
"""Compare two Attention benchmark JSON files offline; no torch or GPU dependency.

speedup = baseline_ms / candidate_ms; values greater than 1 mean the candidate is faster.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import uuid

SCHEMA_VERSION = 1
METRICS = ("fwd", "bwd", "fwd_bwd")
TIMERS = ("cuda_event", "wall")
ESTIMATOR = {"name": "middle2_mean", "sample_count": 6, "drop_fastest": 2, "drop_slowest": 2}
ACCURACY_POLICY = {
    "version": 1, "scope": "full", "forward_reference": "same_dtype_math",
    "gradient_reference": "fp64_math_cast_to_input_dtype",
    "tolerances": {"fp32": {"atol": 1e-4, "rtol": 1e-4},
                   "bf16": {"atol": 2e-2, "rtol": 2e-2},
                   "fp16": {"atol": 5e-3, "rtol": 5e-3}},
    "max_query_chunk": 1024, "max_score_elements": 8 * 1024 * 1024,
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"JSON does not allow non-finite values: {value}")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_object,
                         parse_constant=_bad_constant)


def write_json(path, value, exclusive=False):
    """Reject overwrites on initial writes; atomically replace only results created by this run thereafter."""
    path = Path(path)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
        return
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values):
    if not values:
        return None
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Timing samples must be finite positive numbers")
    mean = statistics.mean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    ordered = sorted(values)
    kept = ordered[2:4] if len(values) == ESTIMATOR["sample_count"] else []
    middle_mean = statistics.mean(kept) if kept else None
    middle_cv = statistics.stdev(kept) / middle_mean if kept else None
    return {"n": len(values), "middle2_mean_ms": middle_mean, "middle2_cv": middle_cv,
            "retained_ms": kept, "dropped_fastest_ms": ordered[:2] if kept else [],
            "dropped_slowest_ms": ordered[4:] if kept else [], "estimator": ESTIMATOR,
            "median_ms": statistics.median(values),
            "mean_ms": mean, "min_ms": min(values), "max_ms": max(values),
            "p10_ms": percentile(values, 0.1), "p90_ms": percentile(values, 0.9),
            "stddev_ms": stddev, "cv": stddev / mean}


def index_records(document, variant):
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Incompatible schema_version")
    if document.get("kind") != "attention_benchmark":
        raise ValueError("Input is not an Attention benchmark result")
    records = {}
    seen = set()
    for record in document["records"]:
        key = (record["case_id"], record["variant"])
        if key in seen:
            raise ValueError(f"Duplicate case/variant: {key}")
        seen.add(key)
        if fingerprint(record["case"]) != record["case_id"]:
            raise ValueError(f"case_id does not match the configuration: {key}")
        if record["variant"] == variant:
            records[record["case_id"]] = record
    if not records:
        raise ValueError(f"No results for variant={variant}")
    return records


def accuracy_status(record, required=False):
    """Accept legacy timing-only data unless accuracy is explicitly required or present."""
    rounds = record.get("rounds", [])
    required = required or record.get("accuracy_required") or any("accuracy" in r for r in rounds)
    if not required:
        return "not_requested"
    if not rounds or len(rounds) != record.get("expected_rounds"):
        return "accuracy_incomplete"
    c = record["case"]
    sizes = {"out": c["b"] * c["hq"] * c["sq"] * c["d"],
             "dq": c["b"] * c["hq"] * c["sq"] * c["d"],
             "dk": c["b"] * c["hkv"] * c["sk"] * c["d"],
             "dv": c["b"] * c["hkv"] * c["sk"] * c["d"]}
    for result in rounds:
        check = result.get("accuracy") or {}
        if check.get("status") not in (None, "ok", "running"):
            return check["status"]
        if check.get("status") != "ok" or check.get("policy") != ACCURACY_POLICY:
            return "accuracy_incomplete"
        tensors = check.get("tensors", {})
        if set(tensors) != set(sizes):
            return "accuracy_incomplete"
        for name, size in sizes.items():
            stats = tensors[name]
            if stats.get("numel") != size:
                return "accuracy_incomplete"
            if (stats.get("status") != "ok" or stats.get("mismatched") != 0
                    or stats.get("nonfinite_actual") != 0 or stats.get("nonfinite_reference") != 0):
                return "accuracy_failed"
            for key in ("max_abs_error", "max_tolerance_ratio"):
                value = stats.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    return "accuracy_incomplete"
            if stats["max_tolerance_ratio"] > 1:
                return "accuracy_failed"
    return "ok"


def _metric_value(record, metric, timer):
    status = accuracy_status(record)
    if status not in ("ok", "not_requested"):
        return None, status
    item = record.get("metrics", {}).get(metric, {})
    status = item.get("status", record.get("status", "runtime_error"))
    if status != "ok":
        return None, status
    rounds = record.get("rounds", [])
    if not rounds or len(rounds) != record.get("expected_rounds"):
        return None, "incomplete_rounds"
    if len({r["round"] for r in rounds}) != len(rounds):
        raise ValueError("Results contain duplicate rounds")
    values = []
    for result in rounds:
        phase = result.get("metrics", {}).get(metric, {})
        if phase.get("status") != "ok":
            return None, phase.get("status", "incomplete_rounds")
        if phase.get("finite") is not True:
            return None, "validation_failed"
        if not phase.get("samples"):
            return None, "no_timing_samples"
        for sample in phase["samples"]:
            value = sample[timer + "_ms"]
            if not math.isfinite(value) or value <= 0 or sample["repeats"] < 1:
                raise ValueError("Invalid raw timing sample")
            values.append(value)
    if len(values) != ESTIMATOR["sample_count"]:
        return None, "requires_exactly_6_samples"
    return summarize(values), "ok"


def speed_summary(rows, tie_pct):
    if not rows:
        return {"paired_count": 0}
    ratios = [r["speedup"] for r in rows]
    changes = [r["latency_change_pct"] for r in rows]
    return {
        "paired_count": len(rows),
        "geomean_speedup": math.exp(statistics.mean(math.log(x) for x in ratios)),
        "median_speedup": statistics.median(ratios),
        "p10_speedup": percentile(ratios, 0.1),
        "p90_speedup": percentile(ratios, 0.9),
        "max_improvement_pct": max(0.0, -min(changes)),
        "max_regression_pct": max(0.0, max(changes)),
        "faster": sum(v < -tie_pct for v in changes),
        "tie": sum(-tie_pct <= v <= tie_pct for v in changes),
        "slower": sum(v > tie_pct for v in changes),
    }


def compare(baseline, candidate, baseline_variant, candidate_variant,
            metric="fwd_bwd", timer="cuda_event", tie_pct=3.0):
    if metric not in METRICS or timer not in TIMERS:
        raise ValueError("Unknown timing metric")
    if not math.isfinite(tie_pct) or tie_pct < 0:
        raise ValueError("tie_pct must be finite and nonnegative")
    left = index_records(baseline, baseline_variant)
    right = index_records(candidate, candidate_variant)
    rows, excluded = [], []
    counts = Counter()
    for case_id in sorted(left.keys() | right.keys()):
        a, b = left.get(case_id), right.get(case_id)
        detail = {"case_id": case_id, "case": (a or b)["case"]}
        if a is None or b is None:
            reason = "candidate_only" if a is None else "baseline_only"
        else:
            if a["case"] != b["case"]:
                raise ValueError(f"Configurations differ for the same case_id: {case_id}")
            av, ast = _metric_value(a, metric, timer)
            bv, bst = _metric_value(b, metric, timer)
            detail.update(baseline_status=ast, candidate_status=bst)
            if ast != "ok" or bst != "ok":
                reason = "unsuccessful"
                counts["baseline_status:" + ast] += 1
                counts["candidate_status:" + bst] += 1
            elif not a.get("inputs") or a["inputs"] != b.get("inputs"):
                reason = "input_mismatch"
            elif a.get("protocol") != b.get("protocol") or not a.get("protocol"):
                reason = "protocol_mismatch"
            else:
                ams, bms = av["middle2_mean_ms"], bv["middle2_mean_ms"]
                detail.update(baseline_ms=ams, candidate_ms=bms,
                              speedup=ams / bms,
                              latency_change_pct=(bms / ams - 1) * 100,
                              baseline_stats=av, candidate_stats=bv)
                rows.append(detail)
                counts["paired"] += 1
                continue
        detail["reason"] = reason
        excluded.append(detail)
        counts[reason] += 1
    grouped = {}
    for field in ("dtype", "mask", "b", "hq", "hkv", "sq", "sk", "d", "layout"):
        buckets = defaultdict(list)
        for row in rows:
            buckets[str(row["case"][field])].append(row)
        grouped[field] = {key: speed_summary(value, tie_pct)
                          for key, value in sorted(buckets.items())}
    def run_info(doc, variant):
        libraries = {}
        for record in doc["records"]:
            if record["variant"] != variant:
                continue
            for result in record.get("rounds", []):
                library = result.get("library")
                if library:
                    libraries[library["sha256"]] = library
        return {"run_id": doc["run_id"], "label": doc.get("label"),
                "variant": variant, "environment": doc.get("environment"),
                "libraries": list(libraries.values()),
                "config": doc.get("config"), "completed": doc.get("completed")}
    return {
        "schema_version": SCHEMA_VERSION, "kind": "attention_comparison",
        "baseline": run_info(baseline, baseline_variant),
        "candidate": run_info(candidate, candidate_variant),
        "metric": metric, "timer": timer, "tie_pct": tie_pct, "estimator": ESTIMATOR,
        "definition": "speedup=baseline/candidate; >1 means the candidate is faster",
        "note": "Each backend uses its default compute settings; +/- thresholds are descriptive, not a test of statistical significance.",
        "coverage": {"union_cases": len(left.keys() | right.keys()),
                     "common_cases": len(left.keys() & right.keys()),
                     "baseline_cases": len(left), "candidate_cases": len(right),
                     **dict(counts)},
        "summary": speed_summary(rows, tie_pct), "groups": grouped,
        "pairs": rows, "excluded": excluded,
    }


def print_comparison(result):
    print(f"Metric={result['metric']} timer={result['timer']}; mean of 6 samples after discarding 2 at each end; >1 means the candidate is faster")
    print("Coverage: " + canonical_json(result["coverage"]))
    print("Overall: " + canonical_json(result["summary"]))
    for field in ("dtype", "mask"):
        print(f"Grouped by {field}:")
        for key, group in result["groups"][field].items():
            print(f"  {key:16s} n={group['paired_count']:4d} "
                  f"geomean_speedup={group['geomean_speedup']:.4f} "
                  f"faster/tie/slower={group['faster']}/{group['tie']}/{group['slower']}")


def validate_device_result(document, variant, device_name):
    records = index_records(document, variant)
    # Accept timing data from legacy both-mode results without running accuracy checks.
    if document.get("config", {}).get("mode") not in ("performance", "both"):
        raise ValueError("Performance comparison requires performance results or legacy both-mode results with timing data")
    devices = set()
    for record in records.values():
        for result in record.get("rounds", []):
            device = (result.get("environment") or {}).get("device", {})
            name = device.get("name", "")
            if not name:
                if result.get("status") == "ok":
                    raise ValueError("A successful record lacks device information; the cross-device baseline cannot be verified")
                continue
            if device_name.lower() not in name.lower():
                raise ValueError(f"Expected {variant} results from {device_name}; actual device: {name}")
            devices.add((name, str(device.get("uuid") or "unknown")))
    if not devices:
        raise ValueError(f"Cannot verify the actual {device_name} device from the results")
    return sorted(devices)


def build_h200_comparisons(baseline, candidate, tie_pct=3.0):
    validate_device_result(baseline, "flex", "H200")
    available = {r["variant"] for r in candidate["records"]}
    variants = [v for v in ("flexflash", "flex") if v in available]
    if not variants:
        raise ValueError("M890 results contain neither flexflash nor flex")
    for variant in variants:
        validate_device_result(candidate, variant, "M890")
    return [compare(baseline, candidate, "flex", variant, metric, timer, tie_pct)
            for variant in variants for metric in METRICS for timer in TIMERS]


def comparison_filename(result):
    return (f"m890_{result['candidate']['variant']}_vs_h200_flex_"
            f"{result['metric']}_{result['timer']}.json")


def markdown_report(results, baseline, candidate, baseline_path, candidate_path):
    """Interpret only paired data; cross-device ratios do not establish kernel bottlenecks or accuracy."""
    def cell(value):
        return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")

    def number(value):
        return "—" if value is None else f"{value:.4f}"

    lines = ["# Attention Cross-Device Performance Comparison", "", "## 1. Comparison Methodology", "",
             f"- H200 baseline JSON: `{cell(baseline_path)}`",
             f"- M890 candidate JSON: `{cell(candidate_path)}`",
             "- The baseline is H200 FlexAttention; the candidate is the corresponding M890 backend.",
             "- speedup = H200 latency / M890 latency; values greater than 1 mean M890 is faster.",
             "- Collect exactly 6 samples per case and direction, sort by latency, discard the fastest 2 and slowest 2, and average the middle 2 before computing ratios (equivalent to the six-sample median).",
             "- Retain all raw samples, medians, untrimmed means, and discarded values; overall speedup is the equally weighted geometric mean across cases.",
             "- Report three directions and two timers separately; summing case latencies does not measure end-to-end training gains.",
             f"- Faster/tie/slower categories use latency changes of +/-{results[0]['tie_pct']:g}%; they do not indicate statistical significance.",
             "", "## 2. Runtime Environment and Sampling Settings", ""]

    def table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
        lines.append("")

    variants = sorted({r["candidate"]["variant"] for r in results})
    for label, doc, selected in (("H200", baseline, ["flex"]), ("M890", candidate, variants)):
        lines.extend([f"### {label}", "", f"- run_id: `{cell(doc.get('run_id'))}`; label: `{cell(doc.get('label'))}`",
                      f"- Collection completed: {doc.get('completed') is True}; sampling configuration: `{cell(canonical_json(doc.get('config', {})))}`"])
        for variant in selected:
            records = index_records(doc, variant)
            envs = [r.get("environment") or {} for record in records.values() for r in record.get("rounds", [])]
            settings = sorted({(str(e.get("torch")), str(e.get("triton")), str(e.get("driver_version")))
                               for e in envs if e})
            devices = validate_device_result(doc, variant, label)
            lines.extend([f"- {variant} actual devices (name, UUID): `{cell(devices)}`",
                          f"- {variant} software versions (torch, triton, driver): `{cell(settings)}`",
                          f"- {variant} case statuses: `{cell(dict(Counter(r['status'] for r in records.values())))}`"])
            libraries = {r["library"]["loaded_path"]: r["library"]["sha256"]
                         for record in records.values() for r in record.get("rounds", []) if r.get("library")}
            for path, digest in libraries.items():
                lines.append(f"- Loaded shared library: `{cell(path)}`; SHA256: `{digest}`")
        lines.append("")

    lines.extend(["## 3. Overall Performance", ""])
    table(["M890 backend", "Direction", "Timer", "Valid/union cases", "Geometric mean speedup", "Median speedup", "p10 / p90", "Faster/tie/slower"],
          [(r["candidate"]["variant"], r["metric"], r["timer"],
            f"{r['summary']['paired_count']}/{r['coverage']['union_cases']}",
            number(r["summary"].get("geomean_speedup")), number(r["summary"].get("median_speedup")),
            f"{number(r['summary'].get('p10_speedup'))} / {number(r['summary'].get('p90_speedup'))}",
            "/".join(str(r["summary"].get(k, 0)) for k in ("faster", "tie", "slower"))) for r in results])

    lines.extend(["## 4. Grouped Forward-and-Backward Results", "",
                  "Results below are grouped by dtype, mask, D, and layout; see the comparison JSON files for all directions and per-case data.", ""])
    for result in results:
        if result["metric"] != "fwd_bwd":
            continue
        lines.extend([f"### M890 {result['candidate']['variant']} / {result['timer']}", ""])
        for field in ("dtype", "mask", "d", "layout"):
            table([field, "Valid cases", "Geometric mean speedup", "Faster/tie/slower"],
                  [(key, group["paired_count"], number(group["geomean_speedup"]),
                    f"{group['faster']}/{group['tie']}/{group['slower']}")
                   for key, group in result["groups"][field].items()])

    lines.extend(["## 5. Cases with the Largest Differences", "", "Using fwd_bwd / cuda_event; latencies are in milliseconds.", ""])
    for result in results:
        if (result["metric"], result["timer"]) != ("fwd_bwd", "cuda_event"):
            continue
        for title, predicate, reverse in (("M890 faster", lambda p: p["latency_change_pct"] < -result["tie_pct"], True),
                                           ("M890 slower", lambda p: p["latency_change_pct"] > result["tie_pct"], False)):
            selected = sorted(filter(predicate, result["pairs"]), key=lambda p: p["speedup"], reverse=reverse)[:5]
            lines.extend([f"### {result['candidate']['variant']}: {title}, up to 5 cases", ""])
            if not selected:
                lines.extend(["No valid cases meet the threshold.", ""])
                continue
            table(["case_id", "dtype / mask", "B,Hq,Hkv,Sq,Sk,D", "layout", "H200 ms", "M890 ms", "Speedup", "H200/M890 CV"],
                  [(p["case_id"][:16], f"{p['case']['dtype']} / {p['case']['mask']}",
                    ",".join(str(p["case"][k]) for k in ("b", "hq", "hkv", "sq", "sk", "d")), p["case"]["layout"],
                    number(p["baseline_ms"]), number(p["candidate_ms"]), number(p["speedup"]),
                    f"{number(p['baseline_stats']['cv'])}/{number(p['candidate_stats']['cv'])}") for p in selected])

    lines.extend(["## 6. Coverage and Exclusion Reasons", "",
                  "Failed or unsupported cases, input hash mismatches, and protocol mismatches are excluded from speedup calculations; exclusion does not imply a performance regression.", ""])
    table(["M890 backend", "Direction", "Timer", "Exclusion reasons and status counts"],
          [(r["candidate"]["variant"], r["metric"], r["timer"],
            canonical_json({k: v for k, v in r["coverage"].items()
                            if k not in ("union_cases", "common_cases", "baseline_cases", "candidate_cases", "paired")}))
           for r in results])
    for label, doc in (("H200", baseline), ("M890", candidate)):
        errors = Counter()
        for record in doc["records"]:
            if record["variant"] not in (["flex"] if label == "H200" else variants):
                continue
            for result in record.get("rounds", []):
                if result.get("status") != "ok":
                    error = result.get("error", {})
                    errors[(record["variant"], result.get("status"), error.get("phase", "unknown"),
                            str(error.get("message", "See error and timing fields in the raw JSON"))[:400])] += 1
        if errors:
            lines.extend([f"### {label} Failure Summary (counted per worker round; messages limited to 400 characters)", ""])
            table(["Backend", "Status", "Phase", "Error message", "Count"], [(*key, n) for key, n in errors.items()])

    lines.extend(["## 7. Conclusions and Limitations", ""])
    for result in results:
        if (result["metric"], result["timer"]) != ("fwd_bwd", "cuda_event"):
            continue
        n = result["summary"]["paired_count"]
        variant = result["candidate"]["variant"]
        if n:
            lines.append(f"- M890 {variant}: across {n}/{result['coverage']['union_cases']} valid cases, "
                         f"the geometric mean H200/M890 latency ratio is {result['summary']['geomean_speedup']:.4f}. "
                         "This conclusion applies only to cases supported by both sides and timed successfully.")
        else:
            lines.append(f"- M890 {variant}: no valid pairs; no performance conclusion can be drawn.")
    if not baseline.get("completed") or not candidate.get("completed"):
        lines.append("- Warning: at least one collection is incomplete; this report does not cover the full matrix.")
    for key in ("rounds", "samples_per_round", "warmup", "target_ms"):
        a, b = baseline.get("config", {}).get(key), candidate.get("config", {}).get(key)
        if a != b:
            lines.append(f"- Warning: {key} mismatch, H200={a}, M890={b}; rerun with identical sampling settings before drawing formal conclusions.")
    if any(r["excluded"] for r in results):
        lines.append("- Excluded cases exist; speedups on the valid subset do not imply that every case is faster.")
    for label, doc in (("H200", baseline), ("M890", candidate)):
        statuses = Counter(accuracy_status(r, required=doc.get("config", {}).get("check_accuracy", False))
                           for r in doc["records"])
        lines.append(f"- {label} numerical accuracy statuses: `{cell(dict(statuses))}`. "
                     "Timing alone does not establish numerical accuracy; see per-round accuracy policies and errors.")
    lines.extend([
                  "- Each backend uses its default compute settings; performance results do not establish the accuracy required to replace memory-efficient attention.",
                  "- cuda_event may include device idle time caused by delayed host submission; wall includes host calls and waits. Their difference is not a direct measure of CPU overhead.",
                  "- This compares workloads across devices and software stacks; latency alone cannot identify MMA, bandwidth, register, or other kernel bottlenecks.",
                  "- For compile failures such as D8 cases, preserve the original error; do not substitute 0ms or classify the failure as a performance regression.",
                  "", "## 8. Traceable Data", ""])
    lines.extend(f"- [{comparison_filename(r)}]({comparison_filename(r)})" for r in results)
    return "\n".join(lines) + "\n"


def write_h200_report(baseline_path, candidate_path, output_dir, tie_pct=3.0):
    baseline, candidate = read_json(baseline_path), read_json(candidate_path)
    results = build_h200_comparisons(baseline, candidate, tie_pct)
    text = markdown_report(results, baseline, candidate, baseline_path, candidate_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for result in results:
        write_json(output / comparison_filename(result), result, exclusive=True)
    with (output / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(text)
    complete = baseline.get("completed") is True and candidate.get("completed") is True
    complete = complete and all(r["summary"]["paired_count"] > 0 and not r["excluded"] for r in results)
    print(f"Generated {len(results)} comparison JSON files; report: {output / 'REPORT.md'}")
    if not complete:
        print("Failed, missing, or incomplete cases exist; the report is preserved and the exit code is nonzero.")
    return 0 if complete else 1


def validate_golden(document, cases=None, protocol=None):
    """Validate local golden data; allow different library fingerprints to evaluate updated kernels."""
    if not isinstance(document, dict):
        raise ValueError("Golden data must be a benchmark JSON object")
    records = index_records(document, "flexflash")
    config = document.get("config", {})
    if document.get("completed") is not True:
        raise ValueError("Golden collection is incomplete")
    if len(document["records"]) != len(records) or config.get("case_count") != len(records):
        raise ValueError("Golden data must contain only flexflash records, with case_count matching the record count")
    if config.get("rounds") != 1 or config.get("samples_per_round") != ESTIMATOR["sample_count"]:
        raise ValueError("Golden data must use 1 round x 6 samples")
    policy = config.get("protocol")
    if not policy or policy.get("estimator") != ESTIMATOR or (protocol is not None and policy != protocol):
        raise ValueError("Golden timing protocol does not match the current method")
    if cases is not None and set(records) != {fingerprint(c) for c in cases}:
        raise ValueError("Golden case list differs from the current configuration; comparing only the common subset is not allowed")
    validate_device_result(document, "flexflash", "M890")
    for case_id, record in records.items():
        if record.get("status") != "ok" or record.get("expected_rounds") != 1:
            raise ValueError(f"Incomplete or failed golden case: {case_id}")
        if record.get("protocol") != policy or not record.get("inputs"):
            raise ValueError(f"Golden case has missing inputs or a protocol mismatch: {case_id}")
        for result in record.get("rounds", []):
            if result.get("status") != "ok" or result.get("round") != 0 or result.get("protocol") != policy:
                raise ValueError(f"Invalid golden worker status or protocol: {case_id}")
        for metric in METRICS:
            for timer in TIMERS:
                _, status = _metric_value(record, metric, timer)
                if status != "ok":
                    raise ValueError(f"Golden {case_id}/{metric}/{timer}: {status}")
    return records


def check_golden(baseline, candidate, require_accuracy=False):
    """Check one-sided performance regression and any requested numerical validation."""
    issues, groups, failed_cases = [], [], set()
    accuracy_checks = []
    try:
        validate_golden(baseline)
        if not isinstance(candidate, dict):
            raise ValueError("Current results must be a benchmark JSON object")
        records = index_records(candidate, "flexflash")
        config = candidate.get("config", {})
        require_accuracy = require_accuracy or config.get("check_accuracy", False)
        if require_accuracy and config.get("accuracy_policy") != ACCURACY_POLICY:
            issues.append("Current accuracy policy is missing or incompatible")
        if candidate.get("completed") is not True:
            issues.append("Current collection is incomplete")
        if len(candidate["records"]) != len(records) or config.get("case_count") != len(records):
            issues.append("Current collection has incomplete backend coverage or case counts")
        if not candidate.get("run_id") or candidate.get("run_id") == baseline.get("run_id"):
            issues.append("Use results from an independent current run; comparing golden data against itself is not allowed")
        validate_device_result(candidate, "flexflash", "M890")
        for key in ("rounds", "samples_per_round", "warmup", "target_ms", "protocol"):
            if config.get(key) is None or config.get(key) != baseline.get("config", {}).get(key):
                issues.append(f"Current sampling setting differs from golden: {key}")
        for case_id, record in records.items():
            accuracy = accuracy_status(record, required=require_accuracy)
            accuracy_checks.append({"case_id": case_id, "status": accuracy})
            if accuracy not in ("ok", "not_requested"):
                failed_cases.add(case_id)
            if record.get("status") != "ok" or record.get("protocol") != config.get("protocol"):
                failed_cases.add(case_id)
            for result in record.get("rounds", []):
                if result.get("status") != "ok" or result.get("protocol") != config.get("protocol"):
                    failed_cases.add(case_id)
        for metric in METRICS:
            for timer in TIMERS:
                result = compare(baseline, candidate, "flexflash", "flexflash", metric, timer)
                regressions = 0
                for pair in result["pairs"]:
                    ratio = pair["speedup"]
                    regressed = ratio < 0.95 and not math.isclose(ratio, 0.95, rel_tol=1e-12)
                    pair.update(performance_change_pct=(ratio - 1) * 100,
                                status="fail" if regressed else "pass")
                    if regressed:
                        regressions += 1
                        failed_cases.add(pair["case_id"])
                failed_cases.update(p["case_id"] for p in result["excluded"])
                groups.append({"metric": metric, "timer": timer, "coverage": result["coverage"],
                               "regression_count": regressions, "pairs": result["pairs"],
                               "excluded": result["excluded"]})
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        issues.append(str(error))
    passed = not issues and not failed_cases and len(groups) == len(METRICS) * len(TIMERS)
    return {"schema_version": SCHEMA_VERSION, "kind": "attention_golden_check",
            "status": "pass" if passed else "fail", "passed": passed, "threshold_pct": 5.0,
            "definition": "performance_ratio=golden_ms/current_ms; fail if ratio<0.95; exactly 5% lower performance or any speedup does not fail",
            "estimator": ESTIMATOR, "accuracy_required": bool(require_accuracy),
            "accuracy_checks": accuracy_checks,
            "baseline_run_id": baseline.get("run_id") if isinstance(baseline, dict) else None,
            "candidate_run_id": candidate.get("run_id") if isinstance(candidate, dict) else None, "issues": issues,
            "failed_case_count": len(failed_cases), "failed_case_ids": sorted(failed_cases), "groups": groups}


def check_repeatability(baseline, candidate, threshold_pct=1.0):
    """Check flexflash only; retain failed and unstable cases instead of filtering them to improve pass rates."""
    if not math.isfinite(threshold_pct) or threshold_pct <= 0:
        raise ValueError("The repeatability threshold must be finite and positive")
    issues = []
    if baseline.get("run_id") == candidate.get("run_id"):
        issues.append("Two independent runs are required; comparing a run against itself is not allowed")
    if baseline.get("config") != candidate.get("config"):
        issues.append("Sampling parameters or protocols differ between runs")
    hashes = [d.get("environment", {}).get("benchmark_sha256") for d in (baseline, candidate)]
    if not hashes[0] or hashes[0] != hashes[1]:
        issues.append("Benchmark source code differs between runs or fingerprints are missing")
    identities = []
    diagnostics = []
    for label, document in (("baseline", baseline), ("candidate", candidate)):
        if document.get("completed") is not True:
            issues.append(f"{label} is incomplete")
        identity = set()
        for record in index_records(document, "flexflash").values():
            for result in record.get("rounds", []):
                env = result.get("environment") or {}
                device = env.get("device", {})
                library = result.get("library") or {}
                identity.add((device.get("uuid"), library.get("sha256"), env.get("torch"),
                              env.get("driver_version"), tuple(env.get("cpu_affinity") or [])))
                for metric in METRICS:
                    phase = result.get("metrics", {}).get(metric, {})
                    reasons = []
                    if phase.get("warmup", {}).get("converged") is not True:
                        reasons.append("warmup_not_converged_or_unverified")
                    if phase.get("calibration", {}).get("target_reached") is not True:
                        reasons.append("calibration_target_not_reached_or_unverified")
                    for timer in TIMERS:
                        if phase.get("stability", {}).get(timer, {}).get("status") != "stable":
                            reasons.append(timer + ":unstable_or_unverified")
                    if reasons:
                        diagnostics.append({"side": label, "case_id": record["case_id"],
                                            "round": result["round"], "metric": metric, "reasons": reasons})
        identities.append(identity)
    if len(identities[0]) != 1 or identities[0] != identities[1] or any(
            not value for row in identities[0] for value in row):
        issues.append("Device, shared library, runtime, or CPU affinity differs, or identifying information is missing")
    groups = []
    for metric in METRICS:
        for timer in TIMERS:
            comparison = compare(baseline, candidate, "flexflash", "flexflash", metric, timer, threshold_pct)
            pairs = comparison["pairs"]
            changes = [abs(p["latency_change_pct"]) for p in pairs]
            over = sum(v > threshold_pct for v in changes)
            groups.append({"metric": metric, "timer": timer, "coverage": comparison["coverage"],
                           "within_threshold": len(pairs) - over, "over_threshold": over,
                           "median_abs_change_pct": statistics.median(changes) if changes else None,
                           "max_abs_change_pct": max(changes) if changes else None,
                           "pairs": pairs, "excluded": comparison["excluded"]})
    matched = all(g["pairs"] and not g["excluded"] for g in groups)
    within = matched and not issues and all(g["over_threshold"] == 0 for g in groups)
    return {"schema_version": SCHEMA_VERSION, "kind": "attention_repeatability",
            "variant": "flexflash", "threshold_pct": threshold_pct,
            "definition": "abs(candidate_middle2_mean_ms/baseline_middle2_mean_ms-1)*100",
            "estimator": ESTIMATOR,
            "baseline_run_id": baseline.get("run_id"), "candidate_run_id": candidate.get("run_id"),
            "within_threshold": within, "passed": within and not diagnostics,
            "issues": issues, "measurement_warnings": diagnostics, "groups": groups,
            "note": "Use 6 samples, discard 2 at each end, and average the middle 2; retain all raw samples. Check within-round stability and cross-run drift separately; future runs are not guaranteed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-variant", choices=("flexflash", "flex"))
    parser.add_argument("--candidate-variant", choices=("flexflash", "flex"))
    report_mode = parser.add_mutually_exclusive_group()
    report_mode.add_argument("--repeatability", action="store_true", help="Check two independent flexflash runs; return 1 on failure")
    report_mode.add_argument("--golden-check", action="store_true", help="flexflash only; return fail/1 if any measurement loses more than 5% performance")
    parser.add_argument("--require-accuracy", action="store_true", help="Require complete output/gradient accuracy checks in the golden candidate")
    parser.add_argument("--threshold-pct", type=float, default=1.0, help="Repeatability threshold; default: 1%")
    report_mode.add_argument("--h200-report", action="store_true",
                        help="baseline=H200, candidate=M890; all directions/timers; output must be a new report directory")
    parser.add_argument("--metric", choices=METRICS, default="fwd_bwd")
    parser.add_argument("--timer", choices=TIMERS, default="cuda_event")
    parser.add_argument("--tie-pct", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.require_accuracy and not args.golden_check:
        parser.error("--require-accuracy requires --golden-check")
    if args.golden_check:
        if args.baseline_variant or args.candidate_variant:
            parser.error("--golden-check checks only flexflash and does not accept variant arguments")
        try:
            result = check_golden(read_json(args.baseline), read_json(args.candidate), args.require_accuracy)
            result["source_files"] = {
                label: {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
                for label, path in (("golden", args.baseline), ("current", args.candidate))}
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = {"schema_version": SCHEMA_VERSION, "kind": "attention_golden_check",
                      "status": "fail", "passed": False, "issues": [str(error)], "groups": []}
        write_json(args.output, result, exclusive=True)
        for group in result["groups"]:
            print(canonical_json({k: v for k, v in group.items() if k not in ("pairs", "excluded")}))
        print(f"Golden check: {result['status']}; JSON: {args.output}")
        return 0 if result["passed"] else 1
    if args.repeatability:
        if args.baseline_variant or args.candidate_variant:
            parser.error("--repeatability checks only flexflash and does not accept variant arguments")
        result = check_repeatability(read_json(args.baseline), read_json(args.candidate), args.threshold_pct)
        write_json(args.output, result, exclusive=True)
        for group in result["groups"]:
            print(canonical_json({k: v for k, v in group.items() if k not in ("pairs", "excluded")}))
        print(f"Repeatability check: {'pass' if result['passed'] else 'fail'}; JSON: {args.output}")
        return 0 if result["passed"] else 1
    if args.h200_report:
        if args.baseline_variant or args.candidate_variant:
            parser.error("--h200-report selects backends automatically and does not accept variant arguments")
        return write_h200_report(args.baseline, args.candidate, args.output, args.tie_pct)
    if not args.baseline_variant or not args.candidate_variant:
        parser.error("Standard comparison requires --baseline-variant and --candidate-variant")
    result = compare(read_json(args.baseline), read_json(args.candidate),
                     args.baseline_variant, args.candidate_variant,
                     args.metric, args.timer, args.tie_pct)
    write_json(args.output, result, exclusive=True)
    print_comparison(result)
    print(f"Comparison JSON: {Path(args.output).resolve()}")


if __name__ == "__main__":
    sys.exit(main())
