#!/usr/bin/env bash
# Validate flexflash output/gradient accuracy and compare performance against the local sibling golden repository.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
CONFIG="$HERE/configs/standard_cases.json"
PYTORCH_ROOT=$(cd -- "$HERE/../../.." && pwd)
GOLDEN="$(dirname -- "$PYTORCH_ROOT")/training_framework_golden_files/torch/torch_2_10_0/flex_flash_attention/890P/flex_flash_attention_golden.json"
DEVICE=0
WARMUP=5
ROUNDS=1
SAMPLES=6
TARGET_MS=200
TIMEOUT=600
OUTPUT_DIR=
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash run_performance.sh [options]
Run the full flexflash suite once, validate numerical accuracy, and compare each case against the local performance golden JSON.
Options:
  --device N             Logical device index; default: 0
  --output-dir DIR       New result directory; defaults to the repository's profile/ directory
  --config FILE          Defaults to all 150 cases in configs/standard_cases.json (Sq and Sk >1024)
  --warmup N             Initial warmup calls; default: 5, followed by sustained warmup and convergence checks
  --rounds N             Must be 1
  --samples N            Must be 6; sort, discard 2 groups at each end, and average the middle 2
  --target-ms MS         Target duration per group; default: 200
  --timeout SECONDS      Timeout per worker including accuracy checks; default: 600 seconds
  --dry-run              Expand cases only; no golden checks, device initialization, or reports
  -h, --help             Show this help
Set the PYTHON environment variable to select a Python interpreter.
Every case validates all output/dQ/dK/dV elements after timing; failures or missing accuracy results fail the run.
Forward reference: same-dtype MATH. Gradient reference: FP64 MATH, cast back to the input dtype.
Tolerances (atol/rtol): fp32=0.01/0.02, bf16=0.02/0.02, fp16=0.005/0.01.
References are batch/query-chunked without sampling; accuracy errors and elapsed_ms are stored in performance.json.
Accuracy adds runtime outside performance samples; existing timing-only golden files remain compatible.
Fixed golden path (relative to the PyTorch repository root):
../training_framework_golden_files/torch/torch_2_10_0/arbitray_flash_atention/890P/arbitray_flash_attention_golden.json
No network downloads; missing or incompatible golden data fails immediately. The format matches bench_attention.py raw JSON.
Compare fwd/bwd/fwd_bwd x cuda_event/wall; fail if any golden_ms/current_ms < 0.95.
Exactly 5% lower performance or any speedup does not fail; failed, missing, or incomparable cases do fail.
The output directory must not exist; it stores performance.json, golden.json, comparison.json, and logs.
Live progress shows completed/total, percentage, current case, elapsed time, and approximate ETA.
Updates appear at case start/end and every 10 seconds while waiting; they are saved in performance.log.
ETA is based on completed workers and can vary substantially between different case shapes.
CPU affinity and GC policy are unchanged; each group repeats toward the target duration, then normalizes to per-call latency.
Use 6 groups and discard 2 at each end; retain all raw values, medians, and untrimmed means.
Print STATUS=pass/fail; pass exits with 0 and fail with a nonzero code. No two-run repeatability check is performed.
EOF
}

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --device|--output-dir|--config|--warmup|--rounds|--samples|--target-ms|--timeout)
      if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
        printf 'Missing value for argument %s\n' "$1" >&2; exit 2
      fi
      case "$1" in
        --device) DEVICE=$2 ;;
        --output-dir) OUTPUT_DIR=$2 ;;
        --config) CONFIG=$2 ;;
        --warmup) WARMUP=$2 ;;
        --rounds) ROUNDS=$2 ;;
        --samples) SAMPLES=$2 ;;
        --target-ms) TARGET_MS=$2 ;;
        --timeout) TIMEOUT=$2 ;;
      esac
      shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
unset FLEX_FLASH_ATTENTION_SO_PATH
export TORCH_FLEX_FLASH_SDPA_ENABLED=1
CMD=("$PYTHON" "$HERE/bench_attention.py" --config "$CONFIG" --backends flexflash
     --dtypes fp32,bf16,fp16 --device "$DEVICE" --mode performance --check-accuracy
     --warmup "$WARMUP" --rounds "$ROUNDS" --samples "$SAMPLES"
     --target-ms "$TARGET_MS" --timeout "$TIMEOUT" --label flexflash_golden_check)
if ((DRY_RUN)); then
  exec "${CMD[@]}" --dry-run
fi
FINAL_STATUS=fail
trap 'printf "STATUS=%s\n" "$FINAL_STATUS"' EXIT
if [[ ! -f "$GOLDEN" || ! -r "$GOLDEN" ]]; then
  printf 'Local golden file is missing or unreadable: %s\n' "$GOLDEN" >&2
  exit 1
fi
"${CMD[@]}" --dry-run > /dev/null
if [[ -z "$OUTPUT_DIR" ]]; then
  mkdir -p "$PYTORCH_ROOT/profile"
  OUTPUT_DIR=$(mktemp -d "$PYTORCH_ROOT/profile/attention_perf_golden_$(date -u +%Y%m%dT%H%M%S)_XXXXXX")
else
  mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
  mkdir -- "$OUTPUT_DIR"
fi
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$GOLDEN" "$CONFIG" "$OUTPUT_DIR" "$WARMUP" "$TARGET_MS" <<'PY'
import sys
from pathlib import Path
from bench_attention import PROTOCOL, normalize_case
from compare_attention import read_json, validate_golden, write_json, file_sha256
source, config, output, warmup, target = sys.argv[1:]
baseline = read_json(source)
cases = [normalize_case(c) for c in read_json(config)["cases"]]
validate_golden(baseline, cases, PROTOCOL)
if baseline["config"].get("warmup") != int(warmup) or baseline["config"].get("target_ms") != float(target):
    raise ValueError("Golden warmup/target_ms differ from the current sampling parameters")
write_json(Path(output) / "golden.json", baseline, exclusive=True)
write_json(Path(output) / "golden_source.json",
           {"path": str(Path(source).resolve()), "sha256": file_sha256(source)}, exclusive=True)
print(f"Local golden validated: {source}; {len(cases)} cases")
PY
"$PYTHON" - "$DEVICE" <<'PY'
import sys
import torch
name = torch.cuda.get_device_name(int(sys.argv[1]))
print(f"Device confirmed: {name}")
PY
STATUS=0
printf 'Starting performance collection; live progress is saved in %s/performance.log\n' "$OUTPUT_DIR"
"${CMD[@]}" --output "$OUTPUT_DIR/performance.json" 2>&1 | tee "$OUTPUT_DIR/performance.log" || STATUS=$?
if ((STATUS > 1)); then exit "$STATUS"; fi
REPORT_STATUS=0
printf 'Collection finished; comparing all cases against the local golden.\n'
"$PYTHON" "$HERE/compare_attention.py" --golden-check --require-accuracy \
  --baseline "$OUTPUT_DIR/golden.json" --candidate "$OUTPUT_DIR/performance.json" \
  --output "$OUTPUT_DIR/comparison.json" 2>&1 | tee "$OUTPUT_DIR/comparison.log" || REPORT_STATUS=$?
printf 'Performance results: %s/performance.json\nComparison results: %s/comparison.json\n' "$OUTPUT_DIR" "$OUTPUT_DIR"
if ((STATUS != 0)); then exit "$STATUS"; fi
if ((REPORT_STATUS != 0)); then exit "$REPORT_STATUS"; fi
FINAL_STATUS=pass
