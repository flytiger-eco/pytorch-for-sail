"""Development checks for performance collection, statistics, and golden validation; not a backend functional test suite."""
import argparse
import gc
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import torch
import bench_attention as bench
import compare_attention as comparison
from benchmark_masks import BENCHMARK_MASKS, benchmark_dense_mask, benchmark_mask_mod

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", required=True, help="Benchmark JSON containing exactly one successful flexflash timing record")
parser.add_argument("--output", required=True, help="Self-check result JSON; never overwrite existing files")
args = parser.parse_args()

torch.set_num_threads(1)

# Replay first-group calibration contaminated by fixed overhead without accessing a GPU.
timing_args = SimpleNamespace(warmup=5, samples=6, target_ms=200.0)
timing_calls = []
def synthetic_timing(torch_module, fn, repeats, *extra):
    ms = 5.56 if not timing_calls else repeats * 0.05 + 0.01
    timing_calls.append(repeats)
    return {"repeats": repeats, "cuda_event_total_ms": ms, "wall_total_ms": ms,
            "cuda_event_ms": ms / repeats, "wall_ms": ms / repeats}, fn()

with patch.object(bench, "timed_group", side_effect=synthetic_timing), \
     patch.object(bench, "finite", return_value=True), \
     patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "Event"), \
     patch.object(bench.os, "sched_setaffinity") as bind, patch.object(gc, "disable") as disable_gc:
    measured = bench.measure(torch, lambda: (torch.ones(1),), timing_args)
    bind.assert_not_called()
    disable_gc.assert_not_called()
assert statistics.median(s["wall_total_ms"] for s in measured["samples"]) >= 180.0, \
    "First-call overhead contaminated calibration; measured groups are far shorter than the target"
assert len(measured["samples"]) == timing_args.samples
assert measured["calibration"]["target_reached"] and measured["warmup"]["converged"]
assert all(v["status"] == "stable" for v in measured["stability"].values())
noise = [{"cuda_event_ms": v, "wall_ms": v} for v in [1.0, 1.0, 1.2, 1.2]]
assert all(v["status"] == "unstable" for v in bench.sample_stability(noise).values())
assert all(v["status"] == "insufficient_samples" for v in bench.sample_stability(noise[:2]).values())
values = [100.0, 0.1, 9.0, 4.0, 5.0, 1.0]
original = values[:]
trimmed = comparison.summarize(values)
assert trimmed["retained_ms"] == [4.0, 5.0]
assert trimmed["dropped_fastest_ms"] == [0.1, 1.0]
assert trimmed["dropped_slowest_ms"] == [9.0, 100.0]
assert trimmed["middle2_mean_ms"] == trimmed["median_ms"] == 4.5
assert values == original
assert comparison.summarize([1.0] * 6)["middle2_mean_ms"] == 1.0
assert comparison.summarize([1.0] * 5)["middle2_mean_ms"] is None
assert comparison.summarize([1.0] * 7)["middle2_mean_ms"] is None
assert comparison.summarize([1.0] * 10)["middle2_mean_ms"] is None
for invalid in (0.0, -1.0, float("inf"), float("nan")):
    try:
        comparison.summarize([1.0] * 5 + [invalid])
    except ValueError:
        pass
    else:
        raise AssertionError("Trimming must not hide invalid samples")
outliers = [{"cuda_event_ms": v, "wall_ms": v} for v in [0.1, 0.2, 1, 1, 3, 4]]
for check in bench.sample_stability(outliers, use_middle2=True).values():
    assert check["status"] == "stable" and check["raw_cv_pct"] > 1.0 and check["cv_pct"] == 0.0

def too_short(torch_module, fn, repeats, events):
    return {"repeats": repeats, "wall_total_ms": 0.001, "cuda_event_total_ms": 0.001,
            "wall_ms": 0.001 / repeats, "cuda_event_ms": 0.001 / repeats}, ()
with patch.object(bench, "timed_group", side_effect=too_short) as timer:
    calibration, _ = bench.calibrate(torch, None, 100.0, None)
    assert not calibration["target_reached"]
    assert timer.call_count <= bench.TIMING_POLICY["calibration_steps"] * bench.TIMING_POLICY["calibration_groups"]
    assert calibration["repeats"] <= bench.TIMING_POLICY["max_repeats"]
missing = {"expected_rounds": 1, "rounds": [{"status": "runtime_error", "metrics": {}}],
           "protocol": bench.PROTOCOL}
bench.merge_round(missing)
assert missing["status"] == "runtime_error"

with patch.object(bench.time, "monotonic", return_value=120.0), patch("builtins.print") as printed:
    for completed, state in ((0, "START"), (1, "RUNNING"), (2, "DONE status=timeout")):
        bench.print_progress(completed, 2, 0.0, 100.0, "dense fp32 flexflash Sq=4096", state)
lines = [c.args[0] for c in printed.call_args_list]
assert "[0/2   0.0%] START" in lines[0] and "ETA~pending" in lines[0]
assert "[1/2  50.0%] RUNNING" in lines[1] and "ETA~0:02:00" in lines[1]
assert "[2/2 100.0%] DONE status=timeout" in lines[2] and "ETA~0:00:00" in lines[2]
assert all("case_elapsed=0:00:20 elapsed=0:02:00" in line for line in lines)
assert all(c.kwargs.get("flush") is True for c in printed.call_args_list)


def synthetic_process(clock, runtime, code=0):
    process = Mock(returncode=code)
    deadline = clock[0] + runtime
    def wait(timeout):
        clock[0] = min(deadline, clock[0] + timeout)
        if clock[0] < deadline:
            raise subprocess.TimeoutExpired("synthetic-worker", timeout)
        return code
    process.wait.side_effect = wait
    return process


for runtime, timeout, code, expected_waits in (
        (3.0, 25.0, 0, [10.0]), (23.0, 25.0, 1, [10.0, 10.0, 5.0]),
        (40.0, 25.0, 0, [10.0, 10.0, 5.0]), (40.0, 5.0, 0, [5.0])):
    clock = [0.0]
    process = synthetic_process(clock, runtime, code)
    heartbeat = Mock()
    with patch.object(bench.time, "monotonic", side_effect=lambda: clock[0]):
        try:
            assert bench.wait_worker(process, timeout, heartbeat) == code
            assert runtime <= timeout
        except subprocess.TimeoutExpired:
            assert runtime > timeout and clock[0] == timeout
    assert [c.kwargs["timeout"] for c in process.wait.call_args_list] == expected_waits
    assert heartbeat.call_count == len(expected_waits) - 1

mask_checks = 0
for s in (259, 16384):
    q = torch.tensor([0, 1, 31, 63, 128, s - 1]).view(-1, 1)
    k = torch.arange(s).view(1, -1)
    chunk, doc = max(1, s // 8), s // 16
    expected = {
        "dense": torch.ones(q.numel(), s, dtype=torch.bool),
        "causal": k <= q,
        "sliding_window": (k >= q - s // 4) & (k <= q + s // 4),
        "sliding_causal": (k <= q) & (k >= q - 511),
        "chunked_causal": (k <= q) & (k // chunk == q // chunk),
        "blockwise": k // chunk <= q // chunk,
        "prefix_lm": (k <= q) | (k < 512),
        "document": q // doc == k // doc,
        "causal_doc": (q // doc == k // doc) & (k <= q),
        "stair": (k >= ((q // 2048 - 5) * 2048).clamp_min(0)) & (k <= q),
    }
    for name in BENCHMARK_MASKS:
        assert torch.equal(benchmark_mask_mod(name, s, s)(0, 0, q, k), expected[name]), name
        mask_checks += 1

# Compare chunked references with independent full MATH graphs on CPU, including GQA and masked rows.
def full_math_result(case, tensors):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q, k, v, grad = tensors
    mask = benchmark_dense_mask(case["mask"], case["sq"], case["sk"], case["mask_params"], q.device)
    rep = case["hq"] // case["hkv"]
    def forward(qr, kr, vr):
        with sdpa_kernel([SDPBackend.MATH]):
            return torch.nn.functional.scaled_dot_product_attention(
                qr, kr.repeat_interleave(rep, dim=1), vr.repeat_interleave(rep, dim=1), attn_mask=mask)
    with torch.no_grad():
        output = forward(q, k, v)
    inputs = [t.detach().double().requires_grad_(True) for t in (q, k, v)]
    grads = torch.autograd.grad(forward(*inputs), inputs, grad.double())
    return (output, *(g.to(q.dtype) for g in grads))


accuracy_reference_checks = 0
for index, mask in enumerate(BENCHMARK_MASKS):
    for dtype in bench.DTYPES:
        case = bench.normalize_case(dict(mask=mask, b=2, hq=4, hkv=2 if index % 2 else 4,
                                         sq=33, sk=35, d=8, dtype=dtype, layout=bench.LAYOUTS[index % 3]))
        tensors, _ = bench.make_inputs(torch, case, "cpu")
        actual = full_math_result(case, tensors)
        with patch.dict(bench.ACCURACY_POLICY, max_query_chunk=7, max_score_elements=4 * 35 * 5):
            checked = bench.check_accuracy(torch, case, tensors, actual)
            assert checked["status"] == "ok", (case, checked)
            assert checked["query_chunk"] == 5
            record = {"case": case, "expected_rounds": 1, "rounds": [{"accuracy": checked}]}
            assert comparison.accuracy_status(record, required=True) == "ok"
        assert all(t.grad is None for t in tensors)
        accuracy_reference_checks += 1

for index, name in enumerate(("out", "dq", "dk", "dv")):
    for value in (100.0, float("nan"), float("inf")):
        corrupted = [t.clone() for t in actual]
        corrupted[index][-1, -1, -1, -1] = value
        checked = bench.check_accuracy(torch, case, tensors, corrupted)
        assert checked["status"] == "accuracy_failed"
        assert checked["tensors"][name]["mismatched"] >= 1
        comparison.canonical_json(checked)

for value, passed in ((0.5, True), (0.50001, False), (float("nan"), False), (float("inf"), False)):
    stats = dict(status="ok", numel=0, mismatched=0, nonfinite_actual=0,
                 nonfinite_reference=0, max_abs_error=0.0, max_tolerance_ratio=0.0)
    bench.update_accuracy_stats(torch, stats, torch.tensor([value], dtype=torch.float64),
                                torch.zeros(1, dtype=torch.float64), {"atol": 0.5, "rtol": 0.1})
    assert (stats["status"] == "ok") is passed
    comparison.canonical_json(stats)
stats = dict(status="ok", numel=0, mismatched=0, nonfinite_actual=0,
             nonfinite_reference=0, max_abs_error=0.0, max_tolerance_ratio=0.0)
bench.update_accuracy_stats(torch, stats, torch.zeros(1), torch.tensor([float("nan")]),
                            {"atol": 0.5, "rtol": 0.1})
assert stats["nonfinite_reference"] == stats["mismatched"] == 1
comparison.canonical_json(stats)

# Check timing-only workers using CPU operations and mocked timing, without device access.
worker_case = bench.normalize_case(dict(mask="dense", b=1, hq=1, hkv=1, sq=2, sk=2, d=2, dtype="fp32"))
worker_args = SimpleNamespace(worker="synthetic.job.json", worker_output="synthetic.worker.json", device=0,
                              check_accuracy=False)
for scenario in ("ok", "nan", "inf", "runtime_error"):
    tensors, hashes = bench.make_inputs(torch, worker_case, "cpu")
    factor = float(scenario) if scenario in ("nan", "inf") else 1.0
    def synthetic_forward():
        return tensors[0] * (tensors[1] + tensors[2]) * factor
    with patch.object(bench, "read_json", return_value={"case": worker_case, "variant": "flex", "round": 0}), \
         patch.object(bench, "write_json") as saved, \
         patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "set_device"), patch.object(torch.cuda, "synchronize"), \
         patch.object(bench, "runtime_environment", return_value={}), \
         patch.object(bench, "make_inputs", return_value=(tensors, hashes)), \
         patch.object(bench, "setup_backend", return_value=synthetic_forward), \
         patch.object(bench, "measure", return_value=deepcopy(measured)) as timer:
        if scenario == "runtime_error":
            timer.side_effect = RuntimeError("Synthetic timing failure")
        code = bench.execute_worker(worker_args)
        result = saved.call_args.args[1]
    assert code == (0 if scenario == "ok" else 1)
    assert "correctness" not in result
    assert result["status"] == ("validation_failed" if scenario in ("nan", "inf") else scenario)
    assert set(result["metrics"]) == set(comparison.METRICS)
    if scenario == "ok":
        assert timer.call_count == 3
        assert all(len(m["samples"]) == 6 and m["finite"] for m in result["metrics"].values())
    elif scenario in ("nan", "inf"):
        timer.assert_not_called()
        assert all(m["finite"] is False for m in result["metrics"].values())

# Accuracy runs only after all timing samples, and failures must survive worker aggregation.
worker_args.check_accuracy = True
for scenario in ("ok", "accuracy_failed", "accuracy_error"):
    tensors, hashes = bench.make_inputs(torch, worker_case, "cpu")
    def attention_forward():
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.MATH]):
            output = torch.nn.functional.scaled_dot_product_attention(*tensors[:3])
        return output * (2 if scenario == "accuracy_failed" else 1)
    real_check = bench.check_accuracy
    def accuracy_after_timing(*arguments):
        assert timer.call_count == 3
        if scenario == "accuracy_error":
            raise RuntimeError("Synthetic reference failure")
        return real_check(*arguments)
    with patch.object(bench, "read_json", return_value={"case": worker_case, "variant": "flex", "round": 0}), \
         patch.object(bench, "write_json") as saved, \
         patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "set_device"), patch.object(torch.cuda, "synchronize"), \
         patch.object(bench, "runtime_environment", return_value={}), \
         patch.object(bench, "make_inputs", return_value=(tensors, hashes)), \
         patch.object(bench, "setup_backend", return_value=attention_forward), \
         patch.object(bench, "measure", return_value=deepcopy(measured)) as timer, \
         patch.object(bench, "check_accuracy", side_effect=accuracy_after_timing):
        code = bench.execute_worker(worker_args)
        result = saved.call_args.args[1]
    assert code == (0 if scenario == "ok" else 1)
    assert result["accuracy"]["status"] == scenario
    assert all(t.grad is None for t in tensors)
    record = {"case": worker_case, "expected_rounds": 1, "rounds": [result],
              "accuracy_required": True, "protocol": bench.PROTOCOL}
    bench.merge_round(record)
    assert (record["status"] == "ok") == (scenario == "ok")
    assert all(m["status"] == "ok" for m in record["metrics"].values())
    comparison.canonical_json(record)

baseline = comparison.read_json(args.baseline)
baseline["records"] = [r for r in baseline["records"] if r["variant"] == "flexflash"]
if len(baseline["records"]) != 1 or baseline["records"][0]["status"] != "ok":
    raise ValueError("The self-check baseline requires exactly one successful flexflash record, such as a single-case pilot.json")
# The historical pilot supplies only input/environment structure; statistics use explicitly labeled synthetic six-sample data.
baseline["label"] = "synthetic-six-samples-not-hardware-data"
for record in baseline["records"]:
    record["rounds"] = record["rounds"][:1]
    record["expected_rounds"] = 1
    for phase in record["rounds"][0]["metrics"].values():
        phase["samples"] = [deepcopy(phase["samples"][0]) for _ in range(6)]
        for sample, value in zip(phase["samples"], values):
            sample.update(cuda_event_ms=value, wall_ms=value,
                          cuda_event_total_ms=value * sample["repeats"],
                          wall_total_ms=value * sample["repeats"])
candidate = deepcopy(baseline)
candidate["label"] = "synthetic-half-time-not-hardware-data"
for record in candidate["records"]:
    record["variant"] = "flex"
    for result in record["rounds"]:
        for phase in result["metrics"].values():
            for sample in phase["samples"]:
                for key in ("cuda_event_ms", "wall_ms", "cuda_event_total_ms", "wall_total_ms"):
                    sample[key] /= 2
for metric in comparison.METRICS:
    for timer in comparison.TIMERS:
        result = comparison.compare(baseline, candidate, "flexflash", "flex", metric, timer)
        assert result["summary"]["geomean_speedup"] == 2.0
        assert result["pairs"][0]["latency_change_pct"] == -50.0
        assert result["pairs"][0]["baseline_ms"] == 4.5
bad = deepcopy(candidate)
bad["records"][0]["rounds"][0]["metrics"]["fwd_bwd"]["samples"].pop()
assert comparison.compare(baseline, bad, "flexflash", "flex")["coverage"]["candidate_status:requires_exactly_6_samples"] == 1
bad = deepcopy(candidate)
bad["records"][0]["metrics"]["fwd_bwd"]["status"] = "validation_failed"
assert comparison.compare(baseline, bad, "flexflash", "flex")["summary"]["paired_count"] == 0
bad = deepcopy(candidate)
bad["records"][0]["inputs"]["tensor_sha256"]["q"] = "different"
assert comparison.compare(baseline, bad, "flexflash", "flex")["coverage"]["input_mismatch"] == 1
bad = deepcopy(candidate)
bad["records"].append(deepcopy(bad["records"][0]))
try:
    comparison.compare(baseline, bad, "flexflash", "flex")
except ValueError:
    pass
else:
    raise AssertionError("Duplicate cases were not rejected")
# The following device names and timings are synthetic test data, not actual H200 performance.
h200 = deepcopy(baseline)
h200["label"] = "synthetic-h200-not-hardware-data"
h200["config"]["mode"] = "performance"
for record in h200["records"]:
    record["variant"] = "flex"
    for result in record["rounds"]:
        result["environment"]["device"]["name"] = "SYNTHETIC H200"
m890 = deepcopy(candidate)
m890["config"]["mode"] = "performance"
m890["records"].append(deepcopy(m890["records"][0]))
m890["records"][1]["variant"] = "flexflash"
for record in m890["records"]:
    for result in record["rounds"]:
        result["environment"]["device"]["name"] = "SYNTHETIC M890"
comparisons = comparison.build_h200_comparisons(h200, m890)
assert len(comparisons) == 12
assert all(r["summary"]["geomean_speedup"] == 2.0 for r in comparisons)
text = comparison.markdown_report(comparisons, h200, m890, "synthetic-h200.json", "synthetic-m890.json")
assert "2.0000" in text and "does not establish numerical accuracy" in text
assert "Accuracy check rounds" not in text
assert "fastest 2" in text and "slowest 2" in text and "middle 2" in text
with patch.object(comparison, "read_json", side_effect=[h200, m890]), \
     patch.object(comparison.Path, "mkdir"), patch.object(comparison.Path, "open", mock_open()) as output, \
     patch.object(comparison, "write_json") as json_output:
    assert comparison.write_h200_report("synthetic-h200.json", "synthetic-m890.json", "synthetic-report") == 0
    assert json_output.call_count == 12
    assert "# Attention Cross-Device Performance Comparison" in output().write.call_args.args[0]
failed = deepcopy(m890)
for record in failed["records"]:
    record["status"] = "compile_error"
    for metric in record["metrics"].values():
        metric["status"] = "compile_error"
    for result in record["rounds"]:
        result.update(status="compile_error", error={"phase": "compile", "message": "SYNTHETIC NYI D8"})
zero_pairs = comparison.build_h200_comparisons(h200, failed)
assert all(r["summary"]["paired_count"] == 0 for r in zero_pairs)
text = comparison.markdown_report(zero_pairs, h200, failed, "synthetic-h200.json", "synthetic-failed.json")
assert "no valid pairs" in text and "SYNTHETIC NYI D8" in text
with patch.object(comparison, "read_json", side_effect=[h200, failed]), \
     patch.object(comparison.Path, "mkdir"), patch.object(comparison.Path, "open", mock_open()), \
     patch.object(comparison, "write_json") as json_output:
    assert comparison.write_h200_report("synthetic-h200.json", "synthetic-failed.json", "synthetic-report") == 1
    assert json_output.call_count == 12
try:
    comparison.validate_device_result(m890, "flex", "H200")
except ValueError:
    pass
else:
    raise AssertionError("The wrong device was accepted as the H200 baseline")
mismatch = deepcopy(m890)
mismatch["config"]["warmup"] += 1
mismatch["completed"] = False
for record in mismatch["records"]:
    record["inputs"]["mask_sha256"] = "different"
text = comparison.markdown_report(comparison.build_h200_comparisons(h200, mismatch),
                                  h200, mismatch, "synthetic-h200.json", "synthetic-mismatch.json")
assert "input_mismatch" in text and "warmup mismatch" in text and "is incomplete" in text

# Repeatability checks must reject self-comparison, threshold violations, and missing cases.
repeat_a = deepcopy(baseline)
repeat_a["completed"] = True
for record in repeat_a["records"]:
    for result in record["rounds"]:
        result["environment"]["cpu_affinity"] = [2, 3]
        for phase in result["metrics"].values():
            phase.update(deepcopy(measured))
repeat_b = deepcopy(repeat_a)
repeat_b["run_id"] += "-independent"
assert comparison.check_repeatability(repeat_a, repeat_b)["passed"]
assert not comparison.check_repeatability(repeat_a, repeat_a)["passed"]
changed = deepcopy(repeat_b)
changed["records"][0]["rounds"][0]["environment"]["cpu_affinity"] = [4, 5]
assert not comparison.check_repeatability(repeat_a, changed)["passed"]
changed = deepcopy(repeat_b)
for result in changed["records"][0]["rounds"]:
    for sample in result["metrics"]["fwd"]["samples"]:
        sample["cuda_event_ms"] *= 1.02
        sample["wall_ms"] *= 1.02
assert not comparison.check_repeatability(repeat_a, changed)["within_threshold"]
changed = deepcopy(repeat_b)
changed["records"][0]["rounds"].pop()
assert not comparison.check_repeatability(repeat_a, changed)["passed"]
changed = deepcopy(repeat_b)
changed["records"][0]["protocol"] = {"version": "synthetic-mismatch"}
assert not comparison.check_repeatability(repeat_a, changed)["within_threshold"]
changed = deepcopy(repeat_b)
changed["records"][0]["rounds"][0]["metrics"]["fwd"]["warmup"]["converged"] = False
assert not comparison.check_repeatability(repeat_a, changed)["passed"]

golden = deepcopy(repeat_a)
golden["config"].update(mode="performance", rounds=1, samples_per_round=6, case_count=1,
                        backends=["flexflash"], protocol=bench.PROTOCOL, warmup=5, target_ms=200.0)
for record in golden["records"]:
    record["protocol"] = bench.PROTOCOL
    for result in record["rounds"]:
        result["protocol"] = bench.PROTOCOL
        result["environment"]["device"]["name"] = "SYNTHETIC M890"
        for phase in result["metrics"].values():
            for sample in phase["samples"]:
                for timer in comparison.TIMERS:
                    sample[timer + "_ms"] = 19.0
                    sample[timer + "_total_ms"] = 19.0 * sample["repeats"]
comparison.validate_golden(golden, [golden["records"][0]["case"]], bench.PROTOCOL)

for scenario in ("ok", "runtime_error", "timeout", "interrupt"):
    with tempfile.TemporaryDirectory(prefix="attention-progress-") as directory:
        clock = [0.0]
        cases = [deepcopy(golden["records"][0]["case"]) for _ in range(2)]
        cases[1]["seed"] += 1
        job_args = SimpleNamespace(output=str(Path(directory) / "run.json"), rounds=1, label="synthetic-progress",
                                   mode="performance", samples=6, warmup=5, target_ms=200.0, timeout=25.0, device=0,
                                   check_accuracy=False)
        def launch(command, **kwargs):
            result = deepcopy(golden["records"][0]["rounds"][0])
            if scenario in ("runtime_error", "timeout", "interrupt"):
                result.update(status="runtime_error" if scenario == "runtime_error" else "running", metrics={})
            comparison.write_json(command[command.index("--worker-output") + 1], result)
            process = synthetic_process(clock, 40.0 if scenario == "timeout" else 23.0,
                                        1 if scenario == "runtime_error" else 0)
            if scenario == "interrupt":
                process.wait.side_effect = KeyboardInterrupt
            return process
        with patch.object(bench, "source_environment", return_value={}), \
             patch.object(bench.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(bench.subprocess, "Popen", side_effect=launch) as launched, \
             patch.object(bench, "stop_worker") as stopped, patch("builtins.print") as printed:
            try:
                assert bench.run_jobs(job_args, cases, ["flexflash"]) == (0 if scenario == "ok" else 1)
                assert scenario != "interrupt"
            except KeyboardInterrupt:
                assert scenario == "interrupt"
        document = comparison.read_json(job_args.output)
        lines = [c.args[0] for c in printed.call_args_list]
        starts = [line for line in lines if "] START |" in line]
        running = [line for line in lines if "] RUNNING |" in line]
        done = [line for line in lines if "] DONE status=" in line]
        assert "[0/2   0.0%]" in starts[0]
        assert "Sq=" in starts[0] and "Hkv=" in starts[0] and "flexflash" in starts[0]
        assert all(c.kwargs.get("flush") is True for c in printed.call_args_list)
        if scenario == "interrupt":
            assert not document["completed"] and not done and not running
            assert launched.call_count == stopped.call_count == 1
        else:
            assert document["completed"] and len(starts) == len(done) == launched.call_count == 2
            assert len(running) == 4 and "[1/2  50.0%]" in starts[1]
            assert "[2/2 100.0%]" in done[-1] and f"DONE status={scenario}" in done[-1]
            assert document["status_counts"] == {scenario: 2}
            assert stopped.call_count == (2 if scenario == "timeout" else 0)


def golden_candidate(ratio):
    current = deepcopy(golden)
    current["run_id"] += "-new"
    current["config"].pop("accuracy_scope", None)
    for result in current["records"][0]["rounds"]:
        result.pop("correctness", None)
        for phase in result["metrics"].values():
            for sample in phase["samples"]:
                for key in ("cuda_event_ms", "wall_ms", "cuda_event_total_ms", "wall_total_ms"):
                    sample[key] /= ratio
    return current


for ratio, passed in ((1.2, True), (1.0, True), (1 / 1.05, True), (0.95, True), (0.949999, False), (0.9, False)):
    result = comparison.check_golden(golden, golden_candidate(ratio))
    assert result["passed"] is passed, (ratio, result["issues"])
    assert len(result["groups"]) == 6
assert not comparison.check_golden(golden, golden)["passed"]
for invalid in (None, [], {}, {"config": None}):
    assert not comparison.check_golden(invalid, golden_candidate(1.0))["passed"]
    assert not comparison.check_golden(golden, invalid)["passed"]
full_golden = deepcopy(golden)
extra = deepcopy(full_golden["records"][0])
extra["case"]["seed"] += 1
extra["case_id"] = comparison.fingerprint(extra["case"])
full_golden["records"].append(extra)
full_golden["config"]["case_count"] = 2
missing_pair = comparison.check_golden(full_golden, golden_candidate(1.0))
assert not missing_pair["passed"] and all(len(g["excluded"]) == 1 for g in missing_pair["groups"])
for metric in comparison.METRICS:
    for timer in comparison.TIMERS:
        bad = golden_candidate(1.0)
        for sample in bad["records"][0]["rounds"][0]["metrics"][metric]["samples"]:
            sample[timer + "_ms"] /= 0.94
        result = comparison.check_golden(golden, bad)
        assert not result["passed"] and result["failed_case_count"] == 1
        assert sum(g["regression_count"] for g in result["groups"]) == 1
for failure in ("missing", "duplicate", "incomplete", "protocol", "inputs", "worker", "samples", "nan", "zero", "count"):
    bad = golden_candidate(1.0)
    record = bad["records"][0]
    if failure == "missing":
        bad["records"].clear()
    elif failure == "duplicate":
        bad["records"].append(deepcopy(record))
    elif failure == "incomplete":
        bad["completed"] = False
    elif failure == "protocol":
        record["protocol"] = {"version": -1}
    elif failure == "inputs":
        record["inputs"]["mask_sha256"] = "different"
    elif failure == "worker":
        record["rounds"][0]["status"] = "runtime_error"
    elif failure == "samples":
        record["rounds"][0]["metrics"]["fwd"]["samples"].pop()
    elif failure in ("nan", "zero"):
        record["rounds"][0]["metrics"]["fwd"]["samples"][0]["wall_ms"] = float("nan") if failure == "nan" else 0.0
    else:
        bad["config"]["case_count"] = 2
    assert not comparison.check_golden(golden, bad)["passed"], failure
try:
    comparison.validate_golden(golden, [], bench.PROTOCOL)
except ValueError:
    pass
else:
    raise AssertionError("Golden data must include every case required by the current run")

def accuracy_candidate(ratio=1.0):
    current = golden_candidate(ratio)
    current["config"].update(check_accuracy=True, accuracy_policy=deepcopy(bench.ACCURACY_POLICY))
    for record in current["records"]:
        record["accuracy_required"] = True
        c = record["case"]
        checks = {}
        for name in ("out", "dq", "dk", "dv"):
            h, s = (c["hq"], c["sq"]) if name in ("out", "dq") else (c["hkv"], c["sk"])
            checks[name] = dict(status="ok", numel=c["b"] * h * s * c["d"], mismatched=0,
                                nonfinite_actual=0, nonfinite_reference=0, max_abs_error=0.0, max_tolerance_ratio=0.0)
        for result in record["rounds"]:
            result["accuracy"] = dict(status="ok", policy=deepcopy(bench.ACCURACY_POLICY), tensors=checks)
    return current


assert comparison.check_golden(golden, accuracy_candidate(), require_accuracy=True)["passed"]
assert not comparison.check_golden(golden, golden_candidate(1.0), require_accuracy=True)["passed"]
for failure in ("missing", "tensor_missing", "mismatch", "count", "policy", "nan", "reference_error", "timeout"):
    bad = accuracy_candidate()
    record = bad["records"][0]
    result = record["rounds"][0]
    check = result["accuracy"]
    if failure == "missing":
        result.pop("accuracy")
    elif failure == "tensor_missing":
        check["tensors"].pop("dv")
    elif failure == "mismatch":
        check["tensors"]["dk"]["max_tolerance_ratio"] = 1.01
    elif failure == "count":
        check["tensors"]["out"]["numel"] -= 1
    elif failure == "policy":
        check["policy"]["scope"] = "sampled"
    elif failure == "nan":
        check["tensors"]["dq"]["nonfinite_actual"] = 1
    elif failure == "reference_error":
        check["status"] = "accuracy_error"
    else:
        check["status"] = "running"
        result["status"] = "timeout"
    bench.merge_round(record)
    assert record["status"] != "ok", failure
    # Do not trust aggregate status; reject invalid accuracy payloads even with forged OK statuses.
    record["status"] = result["status"] = "ok"
    failed = comparison.check_golden(golden, bad)
    assert not failed["passed"] and failed["failed_case_count"] == 1, failure

with tempfile.TemporaryDirectory(prefix="attention-accuracy-runner-") as directory:
    job_args = SimpleNamespace(output=str(Path(directory) / "run.json"), rounds=1, label="synthetic-accuracy",
                               mode="performance", samples=6, warmup=5, target_ms=200.0, timeout=25.0, device=0,
                               check_accuracy=True)
    def launch_accuracy(command, **kwargs):
        assert "--check-accuracy" in command
        result = deepcopy(accuracy_candidate()["records"][0]["rounds"][0])
        result["accuracy"]["tensors"]["dv"]["mismatched"] = 1
        comparison.write_json(command[command.index("--worker-output") + 1], result)
        return Mock(wait=Mock(return_value=0))
    with patch.object(bench, "source_environment", return_value={}), \
         patch.object(bench.subprocess, "Popen", side_effect=launch_accuracy):
        assert bench.run_jobs(job_args, [golden["records"][0]["case"]], ["flexflash"]) == 1
    document = comparison.read_json(job_args.output)
    assert document["config"]["check_accuracy"] and document["status_counts"] == {"accuracy_failed": 1}

here = Path(__file__).resolve().parent
probe = (here / "run_performance.sh").read_text().split('"$PYTHON" - "$DEVICE" <<\'PY\'\n', 1)[1].split("\nPY", 1)[0]
for device_name in ("SYNTHETIC M890", "SYNTHETIC H200", "SYNTHETIC OTHER"):
    with patch.object(sys, "argv", ["-", "1"]), \
         patch.object(torch.cuda, "get_device_name", return_value=device_name) as query_device, \
         patch("builtins.print") as printed:
        exec(compile(probe, "run_performance.sh:device_probe", "exec"), {})
    query_device.assert_called_once_with(1)
    printed.assert_called_once_with(f"Device confirmed: {device_name}")

# Shell integration tests in an isolated directory: real validation and comparison with mocked timing and device queries.
with tempfile.TemporaryDirectory(prefix="attention-golden-") as directory:
    root = Path(directory)
    entry = root / "pytorch/benchmarks/ppu_benchmarks/flex_flash_attention"
    (entry / "configs").mkdir(parents=True)
    for name in ("bench_attention.py", "compare_attention.py", "benchmark_masks.py", "run_performance.sh"):
        shutil.copyfile(here / name, entry / name)
    comparison.write_json(entry / "configs/standard_cases.json", {"schema_version": 1, "cases": [golden["records"][0]["case"]]})
    golden_file = root / "training_framework_golden_files/torch/torch_2_10_0/arbitray_flash_atention/890P/arbitray_flash_attention_golden.json"
    fixture = root / "current.json"
    comparison.write_json(fixture, golden_candidate(1.0))
    fake_python = root / "fake_python"
    fake_python.write_text("#!" + sys.executable + "\n" + '''import os, pathlib, shutil, sys
args = sys.argv[1:]
if args[0].endswith("bench_attention.py") and "--dry-run" not in args:
    assert "--check-accuracy" in args
    assert os.environ["TORCH_FLEX_FLASH_SDPA_ENABLED"] == "1"
    with open(os.environ["CALL_LOG"], "a") as stream:
        stream.write("collect\\n")
    shutil.copyfile(os.environ["CURRENT_FIXTURE"], args[args.index("--output") + 1])
    sys.exit(int(os.environ.get("COLLECT_CODE", "0")))
if args[0] == "-" and len(args) == 2:
    print("SYNTHETIC M890")
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *args])
''')
    fake_python.chmod(0o700)
    env = {**os.environ, "PYTHON": str(fake_python), "CURRENT_FIXTURE": str(fixture)}
    for scenario in ("missing", "invalid", "mismatch", "pass", "regression", "collection_failure",
                     "accuracy_failure", "accuracy_missing", "accuracy_disabled"):
        env["CALL_LOG"] = str(root / (scenario + ".calls"))
        env["COLLECT_CODE"] = "1" if scenario == "collection_failure" else "0"
        if scenario != "missing":
            stored = deepcopy(golden)
            if scenario == "mismatch":
                stored["config"]["protocol"] = {"version": -1}
            comparison.write_json(golden_file, {} if scenario == "invalid" else stored)
        current = accuracy_candidate(0.9 if scenario == "regression" else 1.0)
        if scenario == "accuracy_failure":
            current["records"][0]["rounds"][0]["accuracy"]["tensors"]["dq"]["mismatched"] = 1
        elif scenario == "accuracy_missing":
            current["records"][0]["rounds"][0].pop("accuracy")
        elif scenario == "accuracy_disabled":
            current = golden_candidate(1.0)
        comparison.write_json(fixture, current)
        output_dir = root / (scenario + "-output")
        run = subprocess.run(["bash", str(entry / "run_performance.sh"), "--output-dir", str(output_dir)],
                             env=env, capture_output=True, text=True)
        assert (run.returncode == 0) == (scenario == "pass"), (scenario, run.stdout, run.stderr)
        assert "STATUS=" + ("pass" if scenario == "pass" else "fail") in run.stdout
        calls = Path(env["CALL_LOG"])
        if scenario in ("missing", "invalid", "mismatch"):
            assert not calls.exists()
        else:
            assert calls.read_text().splitlines() == ["collect"]
            assert comparison.read_json(output_dir / "golden.json") == golden
            assert (output_dir / "comparison.json").is_file()
command = ["bash", str(here / "run_performance.sh")]
run = subprocess.run([*command, "--dry-run"], capture_output=True, text=True, check=True)
document = json.loads(run.stdout)
assert document["backends"] == ["flexflash"] and document["check_accuracy"] is True
expanded = document["cases"]
for invalid in (["--device"], ["--samples", "5"], ["--samples", "7"], ["--samples", "10"], ["--rounds", "2"],
                ["--cpu-affinity", "auto"], ["h200"], ["m890"], ["compare"], ["repeatability"], ["--h200-result", "old.json"]):
    rejected = subprocess.run([*command, "--dry-run", *invalid], capture_output=True)
    assert rejected.returncode == 2
assert len(expanded) == len(comparison.read_json(here / "configs/standard_cases.json")["cases"]) == 150
assert all(min(c["sq"], c["sk"]) > 1024 for c in expanded)
assert {c["mask"] for c in expanded} == set(BENCHMARK_MASKS)
assert {c["dtype"] for c in expanded} == {"fp32", "bf16", "fp16"}
bench_command = [sys.executable, str(here / "bench_attention.py"), "--dry-run"]
for invalid in (["--mode", "correctness"], ["--mode", "both"], ["--accuracy-scope", "full"], ["--reference-chunk", "128"]):
    rejected = subprocess.run([*bench_command, *invalid], capture_output=True)
    assert rejected.returncode == 2
with patch.object(sys, "argv", ["bench_attention.py"]):
    defaults = bench.parse_args()
assert defaults.mode == "performance" and defaults.check_accuracy is False
assert not hasattr(defaults, "accuracy_scope") and not hasattr(defaults, "reference_chunk")
report = {"mask_definition_checks": mask_checks, "accuracy_reference_checks": accuracy_reference_checks,
          "accuracy_fault_injection_checks": "passed", "accuracy_worker_and_runner_checks": "passed",
          "live_progress_checks": "passed", "worker_wait_timeout_checks": "passed",
          "progress_runner_checks": "passed",
          "performance_worker_checks": "passed", "performance_only_cli_checks": "passed",
          "synthetic_statistics_checks": "passed", "synthetic_h200_report_checks": "passed",
          "shell_entrypoint_checks": "passed", "device_probe_checks": "passed", "standard_case_count": len(expanded),
          "timing_calibration_checks": "passed", "repeatability_checks": "passed",
          "middle2_estimator_checks": "passed", "no_cpu_or_gc_mutation": "passed",
          "golden_regression_checks": "passed", "golden_shell_integration_checks": "passed"}
comparison.write_json(args.output, report, exclusive=True)
print(comparison.canonical_json(report))
