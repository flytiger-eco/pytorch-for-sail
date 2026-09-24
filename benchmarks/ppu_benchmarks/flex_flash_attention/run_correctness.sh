#!/usr/bin/env bash
# Validate flexflash output/gradient accuracy only: no performance timing, no golden comparison.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
CONFIG="$HERE/configs/standard_cases.json"
PYTORCH_ROOT=$(cd -- "$HERE/../../.." && pwd)
DEVICE=0
TIMEOUT=600
OUTPUT_DIR=
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash run_correctness.sh [options]
Run the flexflash suite once in correctness-only mode and validate numerical accuracy for every case.
This is the accuracy half of run_performance.sh: performance sampling and the golden comparison are skipped.
Options:
  --device N             Logical device index; default: 0
  --output-dir DIR       New result directory; defaults to the repository's profile/ directory
  --config FILE          Defaults to all cases in configs/standard_cases.json
  --timeout SECONDS      Timeout per worker including accuracy checks; default: 600 seconds
  --dry-run              Expand cases only; no accuracy checks, device initialization, or reports
  -h, --help             Show this help
Set the PYTHON environment variable to select a Python interpreter.
Each case runs one forward/backward pass, a finite check, then validates all output/dQ/dK/dV elements.
Forward reference: same-dtype MATH. Gradient reference: FP64 MATH, cast back to the input dtype.
Tolerances (atol/rtol): fp32=0.01/0.02, bf16=0.02/0.02, fp16=0.005/0.01.
References are batch/query-chunked without sampling; dK/dV are accumulated in FP64.
No timing is collected and no golden file is read; per-case accuracy is stored in correctness.json.
Print STATUS=pass/fail; pass exits with 0 and fail with a nonzero code.
EOF
}

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --device|--output-dir|--config|--timeout)
      if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
        printf 'Missing value for argument %s\n' "$1" >&2; exit 2
      fi
      case "$1" in
        --device) DEVICE=$2 ;;
        --output-dir) OUTPUT_DIR=$2 ;;
        --config) CONFIG=$2 ;;
        --timeout) TIMEOUT=$2 ;;
      esac
      shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
unset FLEX_FLASH_ATTENTION_SO_PATH
export TORCH_FLEX_FLASH_SDPA_ENABLED=1
CMD=("$PYTHON" "$HERE/bench_attention.py" --config "$CONFIG" --backends flexflash
     --dtypes fp32,bf16,fp16 --device "$DEVICE" --mode performance --check-accuracy --skip-timing
     --timeout "$TIMEOUT" --label flexflash_correctness_check)
if ((DRY_RUN)); then
  exec "${CMD[@]}" --dry-run
fi
FINAL_STATUS=fail
trap 'printf "STATUS=%s\n" "$FINAL_STATUS"' EXIT
"${CMD[@]}" --dry-run > /dev/null
"$PYTHON" - "$DEVICE" <<'PY'
import sys
import torch
name = torch.cuda.get_device_name(int(sys.argv[1]))
print(f"Device confirmed: {name}")
PY
if [[ -z "$OUTPUT_DIR" ]]; then
  mkdir -p "$PYTORCH_ROOT/profile"
  OUTPUT_DIR=$(mktemp -d "$PYTORCH_ROOT/profile/attention_correctness_$(date -u +%Y%m%dT%H%M%S)_XXXXXX")
else
  mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
  mkdir -- "$OUTPUT_DIR"
fi
STATUS=0
printf 'Starting correctness collection; live progress is saved in %s/correctness.log\n' "$OUTPUT_DIR"
"${CMD[@]}" --output "$OUTPUT_DIR/correctness.json" 2>&1 | tee "$OUTPUT_DIR/correctness.log" || STATUS=$?
if ((STATUS > 1)); then exit "$STATUS"; fi
printf 'Collection finished; summarizing per-case accuracy.\n'
REPORT_STATUS=0
"$PYTHON" - "$OUTPUT_DIR/correctness.json" <<'PY' || REPORT_STATUS=$?
import sys
from compare_attention import read_json
doc = read_json(sys.argv[1])
records = doc.get("records", [])
passed, failed = [], []
for r in records:
    c = r["case"]
    tag = (f"{c['mask']} {c['dtype']} B={c['b']} Hq={c['hq']} Hkv={c['hkv']} "
           f"Sq={c['sq']} Sk={c['sk']} D={c['d']} {c['layout']}")
    acc = r.get("accuracy", {}).get("status", "missing")
    if r.get("status") == "ok" and acc == "ok":
        passed.append(tag)
        continue
    tensors = {}
    for rnd in r.get("rounds", []):
        tensors = (rnd.get("accuracy") or {}).get("tensors") or tensors
    detail = []
    for name in ("out", "dq", "dk", "dv"):
        s = tensors.get(name, {})
        if s.get("status") != "ok":
            detail.append(f"{name}: mismatched={s.get('mismatched')}/{s.get('numel')} "
                          f"max_abs_err={s.get('max_abs_error')} max_tol_ratio={s.get('max_tolerance_ratio')}")
    failed.append((tag, r.get("status"), acc, detail))
print(f"Total cases: {len(records)}; accuracy passed: {len(passed)}; failed: {len(failed)}")
for tag, status, acc, detail in failed:
    print(f"  FAIL [{status}/{acc}] {tag}")
    for line in detail:
        print(f"        {line}")
sys.exit(0 if not failed else 1)
PY
printf 'Correctness results: %s/correctness.json\n' "$OUTPUT_DIR"
if ((STATUS != 0)); then exit "$STATUS"; fi
if ((REPORT_STATUS != 0)); then exit "$REPORT_STATUS"; fi
FINAL_STATUS=pass
