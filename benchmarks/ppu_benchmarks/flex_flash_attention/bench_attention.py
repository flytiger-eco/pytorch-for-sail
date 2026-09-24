#!/usr/bin/env python3
"""Steady-state Attention benchmark with fixed masks and optional full numerical validation.

Both backends use their default compute settings. flexflash runs only through SDPA; Flex runs independently.
Examples:
  TORCH_FLEX_FLASH_SDPA_ENABLED=1 python bench_attention.py --preset smoke --backends flexflash,flex --output run.json
  python bench_attention.py --preset standard --export-cases cases.json --dry-run
  python bench_attention.py --config cases.json --backends flex --output h200.json
  python compare_attention.py --baseline run.json --candidate h200.json \
      --baseline-variant flexflash --candidate-variant flex --output comparison.json

Configuration format: {"schema_version":1,"cases":[{"mask":"causal","b":1,"hq":2,
"hkv":2,"sq":256,"sk":256,"d":64,"dtype":"fp32","layout":"bhsd_contiguous"}]}.
Filters with --config only select cases; they do not modify them. Exported lists can be edited and reused.
Use --check-accuracy to validate every output and gradient element against MATH after timing.
Forward references use the input dtype; gradient references use FP64 and cast back to the input dtype.
Add --skip-timing for a correctness-only run: performance sampling is skipped and each case does just
the finite check plus --check-accuracy validation (no timing, no golden comparison).
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import platform
from queue import Empty
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid

from compare_attention import (
    ACCURACY_POLICY, ESTIMATOR, METRICS, SCHEMA_VERSION, TIMERS, accuracy_status,
    canonical_json, file_sha256, fingerprint, read_json, summarize, write_json,
)

HERE = Path(__file__).resolve().parent
DTYPES = ("fp32", "bf16", "fp16")
LAYOUTS = ("bhsd_contiguous", "bshd_view", "sbhd_view")
TIMING_POLICY = {"event_reuse": True, "gc_policy": "unchanged", "cpu_affinity_policy": "unchanged",
                 "calibration_groups": 3, "calibration_steps": 8, "max_repeats": 100000,
                 "warmup_min_ms": 1000.0, "warmup_max_ms": 5000.0,
                 "warmup_max_groups": 50, "warmup_window": 5, "warmup_cv_pct": 0.5,
                 "stability_threshold_pct": 1.0}
PROTOCOL = {"version": 4, "mask_reuse": "fixed_warmed_metadata", "cuda_graph": False,
            "fwd": "training_requires_grad", "bwd": "fixed_graph_retain_graph",
            "fwd_bwd": "fresh_forward_autograd_grad", "timing": "grouped_event_and_sync_wall",
            "scale": "1/sqrt(d)", "dropout": 0.0, "timing_policy": TIMING_POLICY, "estimator": ESTIMATOR}


class Unsupported(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def csv_values(text, cast=str):
    return [cast(v.strip()) for v in text.split(",") if v.strip()]


def expected_stride(b, h, s, d, layout):
    if layout == "bhsd_contiguous":
        return [h * s * d, s * d, d, 1]
    if layout == "bshd_view":
        return [s * h * d, d, h * d, 1]
    return [h * d, d, b * h * d, 1]


def normalize_case(raw):
    from benchmark_masks import BENCHMARK_MASK_VERSION, benchmark_mask_params
    fixed = {"mask_version": BENCHMARK_MASK_VERSION, "input_generator": "cpu_randn_fp32_v1",
             "training_mode": "requires_grad", "dropout": 0.0, "scale": None}
    fields = {"mask", "b", "hq", "hkv", "sq", "sk", "d", "dtype", "layout", "seed", "mask_params"}
    derived = {"strides", "scale_value"}
    if raw.keys() - fields - fixed.keys() - derived:
        raise ValueError(f"Unknown case fields: {raw.keys() - fields - fixed.keys() - derived}")
    c = {"layout": "bhsd_contiguous", "seed": 0, **raw}
    for key in ("b", "hq", "hkv", "sq", "sk", "d"):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(c["seed"]) is not int or not 0 <= c["seed"] < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    if c["hq"] % c["hkv"] or c.get("dtype") not in DTYPES or c["layout"] not in LAYOUTS:
        raise ValueError("Invalid dtype/layout or Hq/Hkv ratio")
    for key, value in fixed.items():
        if key in raw and raw[key] != value:
            raise ValueError(f"Unsupported {key}: {raw[key]}")
    c.update(fixed)
    c["mask_params"] = benchmark_mask_params(c["mask"], c["sq"], c["sk"], c.get("mask_params"))
    strides = {"q": expected_stride(c["b"], c["hq"], c["sq"], c["d"], c["layout"]),
               "k": expected_stride(c["b"], c["hkv"], c["sk"], c["d"], c["layout"])}
    strides.update(v=strides["k"], grad_out=strides["q"])
    values = {"strides": strides, "scale_value": 1 / math.sqrt(c["d"])}
    for key, value in values.items():
        if key in raw and raw[key] != value:
            raise ValueError(f"Derived field {key} does not match the configuration; remove it from the input list and expand again")
    c.update(values)
    return c


def select_cases(args):
    from benchmark_masks import BENCHMARK_MASKS
    if args.config or args.preset == "standard":
        config = args.config or HERE / "configs" / "standard_cases.json"
        document = read_json(config)
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Incompatible configuration schema_version")
        raw = document["cases"]
    else:
        shapes = ((1, 2, 2, 1024, 1024, 64, LAYOUTS[0]),)
        raw = [dict(zip(("b", "hq", "hkv", "sq", "sk", "d", "layout"), shape),
                    mask=mask, dtype=dtype) for shape in shapes for mask in BENCHMARK_MASKS for dtype in DTYPES]
    unique = {}
    for item in raw:
        c = normalize_case(item)
        unique[fingerprint(c)] = c
    filters = {"mask": args.masks, "dtype": args.dtypes, "layout": args.layouts,
               "b": args.batch, "hq": args.heads, "hkv": args.kv_heads,
               "sq": args.sq, "sk": args.sk, "d": args.hdim}
    for field, text in filters.items():
        if text is not None:
            values = csv_values(text, int if field in ("b", "hq", "hkv", "sq", "sk", "d") else str)
            unique = {key: c for key, c in unique.items() if c[field] in values}
    if args.case_ids:
        wanted = set(csv_values(args.case_ids))
        unique = {key: c for key, c in unique.items() if key in wanted}
    if not unique:
        raise ValueError("No workloads remain after filtering; edit the --config list to add shapes")
    return list(unique.values())


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def source_environment():
    repos = {}
    paths = list(HERE.parents)
    paths += [path / "third_party/flex-flash-attention" for path in HERE.parents if (path / ".git").exists()]
    for path in paths:
        if (path / ".git").exists():
            repos[str(path)] = {
                "revision": command_output(["git", "-C", str(path), "rev-parse", "HEAD"]),
                "status": command_output(["git", "-C", str(path), "status", "--short"])}
    return {"python": sys.version, "platform": platform.platform(), "repositories": repos,
            "benchmark_sha256": {name: file_sha256(HERE / name) for name in
                                 ("bench_attention.py", "compare_attention.py", "benchmark_masks.py")}}


def runtime_environment(torch, device):
    props = torch.cuda.get_device_properties(device)
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    mapping = visible.split(",")[device] if visible else str(device)
    return {"torch": torch.__version__, "torch_path": torch.__file__, "triton": triton_version,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cuda_build": torch.version.cuda, "hip_build": torch.version.hip,
            "compiler_version": command_output(["nvcc", "--version"]),
            "device": {"name": props.name, "logical_index": device, "visible_device": mapping,
                       "cuda_visible_devices": visible, "uuid": str(getattr(props, "uuid", "")) or None,
                       "capability": [props.major, props.minor], "total_memory": props.total_memory,
                       "sm_or_cu_count": props.multi_processor_count,
                       "free_memory_before": torch.cuda.mem_get_info(device)[0]},
            "driver_version": command_output(["nvidia-smi", "--query-gpu=driver_version",
                                               "--format=csv,noheader", "-i", mapping]),
            "unavailable_note": "null indicates information unavailable in this runtime; the full environment is not collected",
            "compile": {"fullgraph": True, "dynamic": False, "mode": "default",
                        "triton.cudagraphs": False, "flex_backend": "TRITON"}}


def tensor_bytes(torch, tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def make_inputs(torch, c, device, hash_inputs=True):
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[c["dtype"]]
    generator = torch.Generator(device="cpu").manual_seed(c["seed"])
    shapes = [(c["b"], c["hq"], c["sq"], c["d"]),
              (c["b"], c["hkv"], c["sk"], c["d"])]
    tensors, hashes = [], {}
    for name, shape in zip(("q", "k", "v", "grad_out"), (shapes[0], shapes[1], shapes[1], shapes[0])):
        host = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)
        if hash_inputs:
            hashes[name] = hashlib.sha256(tensor_bytes(torch, host)).hexdigest()
        t = host.to(device)
        if c["layout"] == "bshd_view":
            t = t.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
        elif c["layout"] == "sbhd_view":
            t = t.permute(2, 0, 1, 3).contiguous().permute(1, 2, 0, 3)
        if list(t.stride()) != c["strides"][name]:
            # Use explicit physical strides even for singleton dimensions to keep cross-platform identities stable.
            stable = torch.empty_strided(shape, c["strides"][name], dtype=dtype, device=device)
            stable.copy_(t)
            t = stable
        t = t.detach().requires_grad_(name != "grad_out")
        tensors.append(t)
    return tensors, hashes


def mask_identity(torch, c, mod):
    digest = hashlib.sha256()
    allowed = 0
    k = torch.arange(c["sk"]).view(1, -1)
    for start in range(0, c["sq"], 256):
        q = torch.arange(start, min(c["sq"], start + 256)).view(-1, 1)
        tile = mod(0, 0, q, k).expand(q.numel(), c["sk"]).contiguous()
        digest.update(tensor_bytes(torch, tile))
        allowed += int(tile.sum())
    return {"mask_sha256": digest.hexdigest(), "visible_elements": allowed,
            "mask_elements": c["sq"] * c["sk"]}


def loaded_library(requested):
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and "libflex_flash_attention.so" in parts[5]:
            paths.add(str(Path(parts[5]).resolve()))
    if len(paths) != 1:
        raise RuntimeError(f"Cannot identify a unique loaded flexflash library: {sorted(paths)}")
    path = paths.pop()
    if requested and path != str(Path(requested).resolve()):
        raise RuntimeError(f"Requested flexflash library differs from the loaded library: {requested} != {path}")
    return {"requested_path": requested, "loaded_path": path, "sha256": file_sha256(path)}


def setup_backend(torch, c, variant, tensors, device):
    from benchmark_masks import benchmark_dense_mask, benchmark_mask_mod
    q, k, v, grad = tensors
    mod = benchmark_mask_mod(c["mask"], c["sq"], c["sk"], c["mask_params"])
    gqa = c["hq"] != c["hkv"]
    if variant == "flexflash":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        if not hasattr(SDPBackend, "FLEX_FLASH_ATTENTION"):
            raise Unsupported("The current torch build does not include the SDPA flexflash backend")
        if torch.cuda.get_device_capability(device) != (8, 9):
            raise Unsupported("flexflash currently supports only M890; use --backends flex on H200")
        mask = benchmark_dense_mask(c["mask"], c["sq"], c["sk"], c["mask_params"], device)
        def forward():
            with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
                    scale=None, enable_gqa=gqa)
    else:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        block_mask = create_block_mask(mod, None, None, c["sq"], c["sk"], device=device, BLOCK_SIZE=128)
        flex = torch.compile(flex_attention, fullgraph=True, dynamic=False,
                             options={"triton.cudagraphs": False})
        def forward():
            return flex(q, k, v, block_mask=block_mask, scale=None, enable_gqa=gqa,
                        kernel_options={"BACKEND": "TRITON"})
    return forward


def finite(torch, tensors):
    return all(bool(torch.isfinite(t).all().item()) for t in tensors)


def update_accuracy_stats(torch, stats, actual, reference, tolerance):
    """Accumulate full-element errors without writing NaN/Inf into result JSON."""
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("Accuracy tensors must have matching shapes and dtypes")
    actual, reference = actual.detach().double(), reference.detach().double()
    actual_finite, reference_finite = torch.isfinite(actual), torch.isfinite(reference)
    valid = actual_finite & reference_finite
    error = (actual - reference).abs()
    bound = tolerance["atol"] + tolerance["rtol"] * reference.abs()
    stats["numel"] += actual.numel()
    stats["mismatched"] += int((~valid | (error > bound)).sum().item())
    stats["nonfinite_actual"] += int((~actual_finite).sum().item())
    stats["nonfinite_reference"] += int((~reference_finite).sum().item())
    stats["max_abs_error"] = max(stats["max_abs_error"], float(error.masked_fill(~valid, 0).max().item()))
    stats["max_tolerance_ratio"] = max(stats["max_tolerance_ratio"],
                                       float((error / bound).masked_fill(~valid, 0).max().item()))
    stats["status"] = "accuracy_failed" if stats["mismatched"] else "ok"


def check_accuracy(torch, c, tensors, actual):
    """Bound score workspace by batch/query chunks; accumulate all KV gradients in FP64."""
    from benchmark_masks import benchmark_mask_mod
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q, k, v, grad = tensors
    out, dq, dk, dv = actual
    tolerance = ACCURACY_POLICY["tolerances"][c["dtype"]]
    stats = {name: {"status": "ok", "numel": 0, "mismatched": 0,
                    "nonfinite_actual": 0, "nonfinite_reference": 0,
                    "max_abs_error": 0.0, "max_tolerance_ratio": 0.0}
             for name in ("out", "dq", "dk", "dv")}
    chunk = max(1, min(c["sq"], ACCURACY_POLICY["max_query_chunk"],
                       ACCURACY_POLICY["max_score_elements"] // (c["hq"] * c["sk"])))
    mod = benchmark_mask_mod(c["mask"], c["sq"], c["sk"], c["mask_params"])
    keys = torch.arange(c["sk"], device=q.device).view(1, -1)
    rep = c["hq"] // c["hkv"]

    def math_forward(qr, kr, vr, mask):
        if rep != 1:
            kr = kr.repeat_interleave(rep, dim=1)
            vr = vr.repeat_interleave(rep, dim=1)
        with sdpa_kernel([SDPBackend.MATH]):
            return torch.nn.functional.scaled_dot_product_attention(
                qr, kr, vr, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=None)

    for batch in range(c["b"]):
        kb, vb = k[batch:batch + 1].detach(), v[batch:batch + 1].detach()
        k64, v64 = kb.double().requires_grad_(True), vb.double().requires_grad_(True)
        dk64, dv64 = torch.zeros_like(k64), torch.zeros_like(v64)
        for start in range(0, c["sq"], chunk):
            end = min(start + chunk, c["sq"])
            index = (slice(batch, batch + 1), slice(None), slice(start, end))
            qb = q[index].detach()
            queries = torch.arange(start, end, device=q.device).view(-1, 1)
            mask = mod(0, 0, queries, keys).expand(end - start, c["sk"]).contiguous()
            with torch.no_grad():
                ref = math_forward(qb, kb, vb, mask)
            update_accuracy_stats(torch, stats["out"], out[index], ref, tolerance)
            q64 = qb.double().requires_grad_(True)
            ref64 = math_forward(q64, k64, v64, mask)
            gq, gk, gv = torch.autograd.grad(ref64, (q64, k64, v64), grad[index].double())
            update_accuracy_stats(torch, stats["dq"], dq[index], gq.to(q.dtype), tolerance)
            dk64.add_(gk)
            dv64.add_(gv)
            del ref, ref64, gq, gk, gv, q64
        update_accuracy_stats(torch, stats["dk"], dk[batch:batch + 1], dk64.to(k.dtype), tolerance)
        update_accuracy_stats(torch, stats["dv"], dv[batch:batch + 1], dv64.to(v.dtype), tolerance)
    return {"status": "ok" if all(s["status"] == "ok" for s in stats.values()) else "accuracy_failed",
            "policy": ACCURACY_POLICY, "query_chunk": chunk, "tensors": stats}


def timed_group(torch, fn, repeats, events):
    start, end = events
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(repeats):
        outputs = fn()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    event_ms = start.elapsed_time(end)
    sample = {"repeats": repeats, "cuda_event_total_ms": event_ms, "wall_total_ms": wall_ms,
              "cuda_event_ms": event_ms / repeats, "wall_ms": wall_ms / repeats}
    return sample, outputs


def sample_stability(samples, threshold_pct=1.0, use_middle2=False):
    """Check within-round dispersion and half-to-half drift; not a substitute for two independent runs."""
    checks = {}
    for timer in TIMERS:
        values = [s[timer + "_ms"] for s in samples]
        if len(values) < 4 or (use_middle2 and len(values) != ESTIMATOR["sample_count"]):
            checks[timer] = {"status": "insufficient_samples"}
            continue
        middle = len(values) // 2
        stats = summarize(values)
        drift = abs(statistics.median(values[middle:]) / statistics.median(values[:middle]) - 1) * 100
        cv_pct = stats["middle2_cv" if use_middle2 else "cv"] * 100
        score = cv_pct if use_middle2 else max(cv_pct, drift)
        checks[timer] = {"status": "stable" if score <= threshold_pct else "unstable",
                         "scope": "retained_middle2" if use_middle2 else "all_samples",
                         "cv_pct": cv_pct, "raw_cv_pct": stats["cv"] * 100,
                         "raw_half_drift_pct": drift, "threshold_pct": threshold_pct}
    return checks


def calibrate(torch, fn, target_ms, events, repeats=1):
    history = []
    achieved = False
    for _ in range(TIMING_POLICY["calibration_steps"]):
        groups = []
        for _ in range(TIMING_POLICY["calibration_groups"]):
            sample, outputs = timed_group(torch, fn, repeats, events)
            groups.append(sample)
        history.extend(groups)
        durations = [s["wall_total_ms"] for s in groups]
        achieved = min(durations) >= 0.9 * target_ms and (max(durations) <= 1.3 * target_ms or repeats == 1)
        if achieved:
            break
        estimate = math.ceil(repeats * target_ms * 1.05 / max(statistics.median(durations), 1e-6))
        updated = max(1, min(TIMING_POLICY["max_repeats"], repeats * 10, estimate))
        if updated == repeats and repeats == TIMING_POLICY["max_repeats"]:
            break
        repeats = updated
    # The final repeats update is unverified; always use a group size that was actually measured.
    return {"repeats": history[-1]["repeats"], "target_ms": target_ms,
            "target_reached": achieved, "samples": history}, outputs


def measure(torch, fn, args):
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    # Events initialize lazily; perform the first record and synchronization before calibration and timing.
    for event in events:
        event.record()
    events[-1].synchronize()
    for _ in range(args.warmup):
        outputs = fn()
    torch.cuda.synchronize()
    if not finite(torch, outputs):
        return {"status": "validation_failed", "finite": False, "samples": []}
    initial, outputs = calibrate(torch, fn, args.target_ms, events)
    warm = []
    elapsed = 0.0
    converged = False
    for _ in range(TIMING_POLICY["warmup_max_groups"]):
        sample, outputs = timed_group(torch, fn, initial["repeats"], events)
        warm.append(sample)
        elapsed += sample["wall_total_ms"]
        window = warm[-TIMING_POLICY["warmup_window"]:]
        checks = sample_stability(window, TIMING_POLICY["warmup_cv_pct"])
        converged = (len(window) == TIMING_POLICY["warmup_window"]
                     and all(c["status"] == "stable" for c in checks.values()))
        if elapsed >= TIMING_POLICY["warmup_min_ms"] and converged:
            break
        if elapsed >= TIMING_POLICY["warmup_max_ms"]:
            break
    # Sustained warmup can change steady-state speed; verify the group duration again.
    calibration, outputs = calibrate(torch, fn, args.target_ms, events, initial["repeats"])
    samples = []
    for _ in range(args.samples):
        sample, outputs = timed_group(torch, fn, calibration["repeats"], events)
        samples.append(sample)
    ok = finite(torch, outputs)
    return {"status": "ok" if ok else "validation_failed", "finite": ok, "samples": samples,
            "initial_calibration": initial, "calibration": calibration,
            "warmup": {"calls": args.warmup, "elapsed_ms": elapsed, "samples": warm,
                       "converged": converged and elapsed >= TIMING_POLICY["warmup_min_ms"]},
            "stability": sample_stability(samples, TIMING_POLICY["stability_threshold_pct"], use_middle2=True),
            "stats": {timer: summarize([s[timer + "_ms"] for s in samples]) for timer in TIMERS}}


def error_status(error, phase):
    text = str(error).lower()
    if isinstance(error, (Unsupported, NotImplementedError)) or "no available kernel" in text:
        return "unsupported"
    if "out of memory" in text or "outofmemory" in type(error).__name__.lower():
        return "oom"
    if phase == "compile" and "illegal memory" not in text:
        return "compile_error"
    return "runtime_error"


def execute_worker(args):
    job = read_json(args.worker)
    c, variant = job["case"], job["variant"]
    result = {"round": job["round"], "started": now(), "status": "running", "metrics": {},
              "protocol": PROTOCOL}
    phase = "initialization"
    if args.check_accuracy:
        result["accuracy"] = {"status": "pending", "policy": ACCURACY_POLICY}
    def checkpoint():
        write_json(args.worker_output, result)
    try:
        import torch
        if not torch.cuda.is_available():
            raise Unsupported("No CUDA/PPU device is available in the current environment")
        torch.set_num_threads(1)
        torch.cuda.set_device(args.device)
        device = f"cuda:{args.device}"
        result["environment"] = runtime_environment(torch, args.device)
        if variant == "flexflash":
            if not hasattr(torch.nn.attention.SDPBackend, "FLEX_FLASH_ATTENTION"):
                raise Unsupported("The current torch build does not include the SDPA flexflash backend")
            if torch.cuda.get_device_capability(args.device) != (8, 9):
                raise Unsupported("flexflash currently supports only M890; use --backends flex on H200")
        tensors, hashes = make_inputs(torch, c, device)
        from benchmark_masks import benchmark_mask_mod
        mod = benchmark_mask_mod(c["mask"], c["sq"], c["sk"], c["mask_params"])
        result["inputs"] = {"tensor_sha256": hashes, **mask_identity(torch, c, mod)}
        checkpoint()
        phase = "compile" if variant == "flex" else "initialization"
        torch.cuda.synchronize()
        begin = time.perf_counter()
        forward = setup_backend(torch, c, variant, tensors, device)
        # Correctness-only (--skip-timing): capture the ATen dispatch route from
        # the single real forward instead of spending an extra profiled probe
        # forward, so the flexflash kernel runs exactly once per case.
        probe_route = not args.skip_timing and variant == "flexflash"
        if args.skip_timing and variant == "flexflash":
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                fixed = forward()
            route = sorted({e.key for e in prof.key_averages() if "flex_flash_attention" in e.key})
        else:
            fixed = forward()
        torch.cuda.synchronize()
        result["initialization_fwd_ms"] = (time.perf_counter() - begin) * 1000
        if variant == "flexflash":
            if probe_route:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    probe = forward()
                route = sorted({e.key for e in prof.key_averages() if "flex_flash_attention" in e.key})
                del probe
            if not route:
                raise RuntimeError("The actual SDPA flexflash ATen dispatch path was not observed")
            result["aten_route"] = route
            result["library"] = loaded_library(os.environ.get("FLEX_FLASH_ATTENTION_SO_PATH"))
        else:
            result["aten_route"] = ["compiled_flex_attention_triton"]
        q, k, v, grad = tensors
        def fwd():
            return (forward(),)
        def bwd():
            return torch.autograd.grad(fixed, (q, k, v), grad, retain_graph=True)
        def combined():
            o = forward()
            return (o, *torch.autograd.grad(o, (q, k, v), grad))
        functions = {"fwd": fwd, "bwd": bwd, "fwd_bwd": combined}
        if args.skip_timing:
            # Correctness-only: the single forward above and the backward that
            # check_accuracy consumes are the only passes; running fwd/bwd/fwd_bwd
            # here would re-execute the flexflash forward for empty timing samples.
            for name in functions:
                result["metrics"][name] = {"status": "ok", "finite": None, "samples": [],
                                           "initialization_ms": 0.0, "timing_skipped": True}
            checkpoint()
        else:
            for name, fn in functions.items():
                phase = "compile" if variant == "flex" and name == "bwd" else name
                begin = time.perf_counter()
                outputs = fn()
                torch.cuda.synchronize()
                initialization_ms = (time.perf_counter() - begin) * 1000
                phase = name
                if not finite(torch, outputs):
                    result["metrics"][name] = {"status": "validation_failed", "finite": False, "samples": []}
                else:
                    result["metrics"][name] = measure(torch, fn, args)
                result["metrics"][name]["initialization_ms"] = initialization_ms
                checkpoint()
        states = [m["status"] for m in result["metrics"].values()]
        if args.check_accuracy:
            phase = "accuracy"
            result["accuracy"]["status"] = "running"
            checkpoint()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result["accuracy"] = check_accuracy(torch, c, tensors, (fixed.detach(), *bwd()))
            torch.cuda.synchronize()
            result["accuracy"]["elapsed_ms"] = (time.perf_counter() - begin) * 1000
            states.append(result["accuracy"]["status"])
            print("Accuracy: " + canonical_json(result["accuracy"]), flush=True)
        result["status"] = next((s for s in states if s != "ok"), "ok")
    except Exception as error:
        status = error_status(error, phase)
        if phase == "accuracy":
            result["accuracy"].update(status="accuracy_error", error_status=status, message=str(error))
        result.update(status=status, error={"phase": phase, "type": type(error).__name__,
                      "message": str(error), "traceback": traceback.format_exc()[-16000:]})
        for name in METRICS:
            result["metrics"].setdefault(name, {"status": status, "finite": None, "samples": []})
    result["finished"] = now()
    checkpoint()
    print(c["mask"], c["dtype"], variant, result["status"], flush=True)
    return 0 if result["status"] == "ok" else 1


def merge_round(record):
    rounds = record["rounds"]
    complete = len(rounds) == record["expected_rounds"]
    identities = [r.get("inputs") for r in rounds if r.get("inputs")]
    consistent = bool(identities) and all(v == identities[0] for v in identities)
    record["inputs"] = identities[0] if consistent else None
    record["protocol"] = rounds[0].get("protocol", record["protocol"]) if rounds else record["protocol"]
    record["metrics"] = {}
    for name in METRICS:
        phases = [r.get("metrics", {}).get(name, {"status": r.get("status", "runtime_error")}) for r in rounds]
        status = next((p["status"] for p in phases if p["status"] != "ok"), "ok")
        if status == "ok" and not consistent:
            status = "validation_failed"
        if status == "ok" and not complete:
            status = "pending"
        samples = [s for p in phases for s in p.get("samples", [])]
        record["metrics"][name] = {"status": status,
                                  "stats": {t: summarize([s[t + "_ms"] for s in samples]) for t in TIMERS}}
    libraries = [r.get("library", {}).get("sha256") for r in rounds if r.get("library")]
    devices = [r.get("environment", {}).get("device", {}).get("uuid") for r in rounds]
    protocols_match = all(r.get("protocol", record["protocol"]) == record["protocol"] for r in rounds)
    affinities = {tuple(r.get("environment", {}).get("cpu_affinity", [])) for r in rounds}
    if len(set(libraries)) > 1 or len(set(devices)) > 1 or len(affinities) > 1 or not protocols_match:
        for metric in record["metrics"].values():
            if metric["status"] == "ok":
                metric["status"] = "validation_failed"
        record["consistency_error"] = "Rounds used different libraries, devices, CPU affinity settings, or timing protocols"
    statuses = [m["status"] for m in record["metrics"].values()]
    statuses.extend(r.get("status", "runtime_error") for r in rounds)
    accuracy = accuracy_status(record)
    if accuracy != "not_requested":
        record["accuracy"] = {"status": accuracy}
        statuses.append(accuracy)
    record["status"] = next((s for s in statuses if s != "ok"), "ok")


def stop_worker(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_worker(process, timeout, on_progress):
    deadline = time.monotonic() + timeout
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            return process.wait(timeout=min(10.0, remaining))
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                raise
            on_progress()


def print_progress(completed, total, started, job_started, description, status):
    current = time.monotonic()
    elapsed = current - started
    def duration(seconds):
        return str(timedelta(seconds=max(0, int(seconds))))
    eta = duration(elapsed / completed * (total - completed)) if completed else "pending"
    print(f"[{completed}/{total} {100.0 * completed / total:5.1f}%] {status} | {description} | "
          f"case_elapsed={duration(current - job_started)} elapsed={duration(elapsed)} ETA~{eta}", flush=True)


def run_jobs(args, cases, backends):
    output = Path(args.output).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    work = output.with_name(output.stem + "." + run_id)
    rounds = args.rounds
    indexes = {(fingerprint(case), variant):
               {"case_id": fingerprint(case), "case": case, "variant": variant,
                "expected_rounds": rounds, "rounds": [], "status": "pending", "protocol": PROTOCOL,
                "accuracy_required": args.check_accuracy}
               for case in cases for variant in backends}
    document = {"schema_version": SCHEMA_VERSION, "kind": "attention_benchmark", "run_id": run_id,
                "label": args.label, "started": now(), "completed": False,
                "environment": source_environment(), "records": list(indexes.values()),
                "config": {"mode": args.mode, "check_accuracy": args.check_accuracy,
                           "skip_timing": args.skip_timing,
                           "accuracy_policy": ACCURACY_POLICY if args.check_accuracy else None,
                           "rounds": rounds, "samples_per_round": args.samples, "warmup": args.warmup,
                           "target_ms": args.target_ms, "timeout_s": args.timeout,
                           "backends": backends, "case_count": len(cases), "protocol": PROTOCOL}}
    write_json(output, document, exclusive=True)
    work.mkdir()
    print(f"Workloads {len(cases)} x backends {len(backends)} x rounds {rounds}; output {output}", flush=True)
    jobs_completed = 0
    total_jobs = len(cases) * len(backends) * rounds
    started = time.monotonic()
    try:
        for case in cases:
            case_id = fingerprint(case)
            for round_index in range(rounds):
                order = backends if round_index % 2 == 0 else list(reversed(backends))
                for variant in order:
                    key = (case_id, variant)
                    record = indexes[key]
                    stem = f"{case_id[:16]}.{variant}.r{round_index}"
                    job_file, worker_file, log_file = (work / (stem + suffix) for suffix in (".job.json", ".json", ".log"))
                    write_json(job_file, {"case": case, "variant": variant, "round": round_index}, exclusive=True)
                    command = [sys.executable, str(HERE / "bench_attention.py"), "--worker", str(job_file),
                               "--worker-output", str(worker_file), "--device", str(args.device),
                               "--mode", args.mode, "--warmup", str(args.warmup),
                               "--samples", str(args.samples), "--target-ms", str(args.target_ms)]
                    if args.check_accuracy:
                        command.append("--check-accuracy")
                    if args.skip_timing:
                        command.append("--skip-timing")
                    env = os.environ.copy()
                    env["TORCH_USE_RTLD_GLOBAL"] = "1"
                    env["TORCHINDUCTOR_CACHE_DIR"] = str(work / "inductor_cache")
                    env["TRITON_CACHE_DIR"] = str(work / "triton_cache")
                    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
                    terminal_status = None
                    description = (f"{case_id[:12]} {case['mask']} {case['dtype']} {variant} r{round_index} "
                                   f"B={case['b']} Hq={case['hq']} Hkv={case['hkv']} "
                                   f"Sq={case['sq']} Sk={case['sk']} D={case['d']} {case['layout']}")
                    job_started = time.monotonic()
                    def progress(status="RUNNING"):
                        print_progress(jobs_completed, total_jobs, started, job_started, description, status)
                    progress("START")
                    with log_file.open("x") as log:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                   env=env, start_new_session=True)
                        try:
                            code = wait_worker(process, args.timeout, progress)
                        except subprocess.TimeoutExpired:
                            stop_worker(process)
                            code, terminal_status = process.returncode, "timeout"
                        except BaseException:
                            stop_worker(process)
                            raise
                    result = read_json(worker_file) if worker_file.exists() else {
                        "round": round_index, "metrics": {}, "status": "runtime_error"}
                    if terminal_status or result.get("status") == "running" or (code != 0 and result.get("status") == "ok"):
                        result["status"] = terminal_status or "runtime_error"
                        result["error"] = {"message": "Worker timed out or exited abnormally", "returncode": code}
                        for name in METRICS:
                            result["metrics"].setdefault(name, {"status": result["status"], "finite": None, "samples": []})
                    result.update(log_path=str(log_file), returncode=code)
                    record["rounds"].append(result)
                    merge_round(record)
                    if document["environment"].get("runtime") is None:
                        document["environment"]["runtime"] = result.get("environment")
                    write_json(output, document)
                    jobs_completed += 1
                    progress(f"DONE status={result['status']}")
        document["completed"] = True
    finally:
        document["finished"] = now()
        document["status_counts"] = dict(Counter(r["status"] for r in document["records"]))
        write_json(output, document)
    print("Results: " + canonical_json(document["status_counts"]), flush=True)
    return 0 if all(r["status"] == "ok" for r in document["records"]) else 1


def _correctness_case(torch, c, variant, dev, verify_route, hash_inputs=False):
    """Run one case's forward+backward exactly once and validate accuracy.

    Correctness-only helper: no timing and no per-case environment/library
    probing. The flexflash ATen dispatch route is verified with a profiler only
    when verify_route is set (once per worker process), so the forward under test
    is never executed more than once per case.
    """
    result = {"round": 0, "started": now(), "status": "running", "metrics": {}, "protocol": PROTOCOL}
    phase = "initialization"
    try:
        tensors, hashes = make_inputs(torch, c, dev, hash_inputs=hash_inputs)
        if hash_inputs:
            result["inputs"] = {"tensor_sha256": hashes}
        forward = setup_backend(torch, c, variant, tensors, dev)
        if verify_route and variant == "flexflash":
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                fixed = forward()
            route = sorted({e.key for e in prof.key_averages() if "flex_flash_attention" in e.key})
            if not route:
                raise RuntimeError("The actual SDPA flexflash ATen dispatch path was not observed")
            result["aten_route"] = route
            result["library"] = loaded_library(os.environ.get("FLEX_FLASH_ATTENTION_SO_PATH"))
        else:
            fixed = forward()
            result["aten_route"] = (["flexflash_dispatch_trusted"] if variant == "flexflash"
                                    else ["compiled_flex_attention_triton"])
        torch.cuda.synchronize()
        q, k, v, grad = tensors
        phase = "accuracy"
        grads = torch.autograd.grad(fixed, (q, k, v), grad)
        accuracy = check_accuracy(torch, c, tensors, (fixed.detach(), *grads))
        torch.cuda.synchronize()
        result["accuracy"] = accuracy
        result["status"] = "ok" if accuracy["status"] == "ok" else accuracy["status"]
    except Exception as error:
        status = error_status(error, phase)
        result["accuracy"] = {"status": "accuracy_error", "error_status": status, "message": str(error)}
        result.update(status=status, error={"phase": phase, "type": type(error).__name__,
                                            "message": str(error), "traceback": traceback.format_exc()[-16000:]})
    result["finished"] = now()
    return result


def _correctness_shard(device, shard, queue):
    """Long-lived worker: import torch once and run a shard of cases in-process."""
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    dev = f"cuda:{device}"
    route_verified = False
    for c, variant in shard:
        record = _correctness_case(torch, c, variant, dev, not route_verified and variant == "flexflash")
        if variant == "flexflash" and record.get("status") == "ok":
            route_verified = True
        queue.put({"case_id": fingerprint(c), "case": c, "variant": variant, "device": device,
                   "expected_rounds": 1, "protocol": PROTOCOL, "accuracy_required": True,
                   "rounds": [record], "status": record["status"],
                   "accuracy": {"status": (record.get("accuracy") or {}).get("status", record["status"])}})
        torch.cuda.empty_cache()


def run_correctness(args, cases, backends):
    """Fastest single-GPU correctness path: run every case exactly once.

    One persistent worker process on args.device runs the whole suite in-process,
    so torch/CUDA/library initialization happens once instead of per case; the
    parent streams records for live progress and writes a run_jobs-compatible
    document. No timing and no golden comparison; pass/fail is per-case accuracy.
    """
    device = args.device
    jobs = [(c, variant) for c in cases for variant in backends]
    total = len(jobs)
    output = Path(args.output).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    document = {"schema_version": SCHEMA_VERSION, "kind": "attention_benchmark", "run_id": run_id,
                "label": args.label, "started": now(), "completed": False,
                "environment": source_environment(), "records": [],
                "config": {"mode": "correctness", "check_accuracy": True, "skip_timing": True,
                           "device": device, "backends": backends, "case_count": len(cases),
                           "job_count": total, "timeout_s": args.timeout,
                           "accuracy_policy": ACCURACY_POLICY, "protocol": PROTOCOL}}
    write_json(output, document, exclusive=True)
    print(f"Correctness: {total} case runs on device {device} in a single persistent worker; output {output}", flush=True)
    ctx = multiprocessing.get_context("spawn")
    started = time.monotonic()
    records = []
    remaining = list(jobs)
    # A single persistent worker runs the whole shard in-process (torch/CUDA/library
    # initialized once). If a case hard-crashes the worker (e.g. an illegal memory
    # access), restart it for the not-yet-done cases; if a restart yields nothing,
    # isolate the leading case as crashed so one poison case cannot block the suite.
    while remaining:
        queue = ctx.Queue()
        proc = ctx.Process(target=_correctness_shard, args=(device, remaining, queue), daemon=True)
        proc.start()
        got, last = 0, time.monotonic()
        try:
            while got < len(remaining):
                try:
                    record = queue.get(timeout=5)
                except Empty:
                    if not proc.is_alive() or time.monotonic() - last > args.timeout:
                        break
                    continue
                last = time.monotonic()
                records.append(record)
                got += 1
                c = record["case"]
                print_progress(len(records), total, started, started,
                               f"{record['case_id'][:12]} {c['mask']} {c['dtype']} {record['variant']} "
                               f"B={c['b']} Sq={c['sq']} Sk={c['sk']} D={c['d']} dev={device}",
                               f"DONE status={record['status']}")
        finally:
            while True:
                try:
                    records.append(queue.get_nowait())
                    got += 1
                except Empty:
                    break
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=5)
        done = {(r["case_id"], r["variant"]) for r in records}
        remaining = [(c, v) for (c, v) in jobs if (fingerprint(c), v) not in done]
        if remaining and got == 0:
            c, variant = remaining[0]
            records.append({"case_id": fingerprint(c), "case": c, "variant": variant, "device": device,
                            "expected_rounds": 1, "protocol": PROTOCOL, "accuracy_required": True,
                            "rounds": [], "status": "worker_crash",
                            "accuracy": {"status": "accuracy_error",
                                         "message": "Worker process died on this case (isolated); rerun it individually"}})
            print_progress(len(records), total, started, started,
                           f"{fingerprint(c)[:12]} {c['mask']} {c['dtype']} {variant}", "CRASH isolated")
    records.sort(key=lambda r: (r["case_id"], r["variant"]))
    all_done = len(records) >= total
    document["records"] = records
    document["completed"] = all_done
    document["finished"] = now()
    document["status_counts"] = dict(Counter(r["status"] for r in records))
    write_json(output, document)
    print("Results: " + canonical_json(document["status_counts"]), flush=True)
    return 0 if all_done and all(r["status"] == "ok" for r in records) else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--config", help="Explicit case list; filters do not modify the configuration")
    parser.add_argument("--export-cases", help="Export an editable case list reusable across devices; never overwrite existing files")
    parser.add_argument("--dry-run", action="store_true", help="Expand the configuration without initializing a GPU")
    parser.add_argument("--backends", default="flexflash,flex")
    parser.add_argument("--dtypes", help="Comma-separated fp32,bf16,fp16; default: all")
    parser.add_argument("--masks", help="Comma-separated mask names; default: all")
    parser.add_argument("--layouts", help="Comma-separated layout names")
    parser.add_argument("--case-ids", help="Comma-separated full case_id values")
    for flag in ("batch", "heads", "kv-heads", "sq", "sk", "hdim"):
        parser.add_argument("--" + flag, help="Comma-separated integers; filter existing cases only")
    parser.add_argument("--device", type=int, default=0, help="Logical device index under the current CUDA_VISIBLE_DEVICES")
    parser.add_argument("--mode", choices=("performance",), default="performance", help="Only performance collection is supported")
    parser.add_argument("--check-accuracy", action="store_true", help="Validate all output/dQ/dK/dV elements against MATH after performance sampling")
    parser.add_argument("--skip-timing", action="store_true", help="Correctness-only: skip performance sampling and run just the finite check plus --check-accuracy validation")
    parser.add_argument("--warmup", type=int, default=5, help="Minimum warmup calls, followed by bounded sustained warmup")
    parser.add_argument("--samples", type=int, default=6, help="Exactly 6 groups; discard the fastest 2 and slowest 2, then average the remaining 2")
    parser.add_argument("--rounds", type=int, default=1, help="Exactly 1 round; repeatability is checked with two independent runs")
    parser.add_argument("--target-ms", type=float, default=200.0)
    parser.add_argument("--timeout", type=float, default=600.0, help="Timeout in seconds per worker")
    parser.add_argument("--label", default="")
    parser.add_argument("--output")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device < 0 or (not args.skip_timing and (min(args.samples, args.rounds) < 1 or args.warmup < 5)):
        parser.error("samples/rounds must be positive, warmup>=5, device>=0")
    if any(not math.isfinite(x) or x <= 0 for x in (args.target_ms, args.timeout)):
        parser.error("target-ms/timeout must be finite positive numbers")
    if not args.skip_timing and (args.samples != 6 or args.rounds != 1):
        parser.error("Performance statistics require 1 round x 6 groups; discard 2 at each end and average the middle 2")
    if args.skip_timing and not args.check_accuracy:
        parser.error("--skip-timing is a correctness-only run and requires --check-accuracy")
    return args


def main():
    args = parse_args()
    if args.worker:
        return execute_worker(args)
    backends = list(dict.fromkeys(csv_values(args.backends)))
    if not backends or set(backends) - {"flexflash", "flex"}:
        raise ValueError("backends must contain only flexflash,flex")
    cases = select_cases(args)
    if args.export_cases:
        editable = [{k: v for k, v in c.items() if k not in ("strides", "scale_value")} for c in cases]
        write_json(args.export_cases, {"schema_version": SCHEMA_VERSION, "cases": editable}, exclusive=True)
    if args.dry_run:
        print(canonical_json({"cases": [{"case_id": fingerprint(c), **c} for c in cases],
                              "case_count": len(cases), "backends": backends,
                              "check_accuracy": args.check_accuracy}))
        return 0
    if not args.output:
        raise ValueError("An actual run requires --output to avoid overwriting historical results")
    if args.skip_timing:
        return run_correctness(args, cases, backends)
    return run_jobs(args, cases, backends)


if __name__ == "__main__":
    sys.exit(main())
