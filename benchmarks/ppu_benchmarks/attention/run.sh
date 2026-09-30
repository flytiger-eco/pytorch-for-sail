#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# PPU Attention Benchmark Runner (golden comparison)
#
# Flow:
#   1. Detect the hardware platform from the CUDA device name.
#   2. Fetch (or update) the training_framework_golden_files repo holding the
#      golden result files and pick the file at
#      torch/<torch_version>/attention/<PLATFORM>/attention_microbenchmark_golden.json,
#      where <torch_version> matches the installed torch (e.g. torch_2_12_0).
#      Golden files are NOT distributed with this repository: the user must
#      prepare them, either via GOLDEN_REPO_URL (a git repository the user can
#      access) or a locally populated GOLDEN_REPO_DIR.
#   3. Run the attention microbenchmark NUM_RUNS times (clearing compiler
#      caches each time) and aggregate the runs with a trimmed mean (drop
#      the TRIM_COUNT fastest and TRIM_COUNT slowest runs, average the
#      rest) to produce a single "current" result.
#   4. Compare the current result against the golden, looking only at latency
#      metrics. If the maximum absolute relative difference exceeds
#      MAX_REL_DIFF_THRESHOLD the script exits non-zero, otherwise it exits 0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

TEST_REPORTS_DIR="${REPO_ROOT}/test/test-reports"
mkdir -p "$TEST_REPORTS_DIR"

NUM_RUNS=10                   # benchmark runs aggregated into the current result
TRIM_COUNT=3                  # drop this many fastest + slowest runs before averaging
METRIC_FILTER="latency"       # only judge latency metrics (forward/backward latency)
MAX_REL_DIFF_THRESHOLD_PCT=5  # fail if max |rel diff| exceeds 5%
MAX_REL_DIFF_THRESHOLD="$(python -c "print(${MAX_REL_DIFF_THRESHOLD_PCT} / 100)")"

# ---------------------------------------------------------------------------
# Step 1: Detect platform from the CUDA device name
# ---------------------------------------------------------------------------
echo "=== Detecting platform ==="
DEVICE_NAME="$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Device name: ${DEVICE_NAME}"
case "$DEVICE_NAME" in
    *M890*) PLATFORM="M890" ;;
    *810E*) PLATFORM="810E" ;;
    *H200*) PLATFORM="H200" ;;
    *)
        echo "ERROR: unrecognized platform from device name '${DEVICE_NAME}'" >&2
        exit 1
        ;;
esac
echo "Detected platform: ${PLATFORM}"

# ---------------------------------------------------------------------------
# Step 2: Fetch the golden-files repo (user-provided) and locate the golden
#         result for this torch version + platform
# ---------------------------------------------------------------------------
# Golden baseline files are NOT distributed with this repository: the user is
# expected to prepare them, either by exporting GOLDEN_REPO_URL to a git
# repository they can access, or by populating GOLDEN_REPO_DIR locally with
# the golden files. No default URL is bundled.
DEFAULT_GOLDEN_REPO_URL=""
GOLDEN_REPO_URL="${GOLDEN_REPO_URL:-$DEFAULT_GOLDEN_REPO_URL}"
# Clone next to the pytorch repo (outside its worktree): nesting a git repo
# inside another repo's worktree triggers "dubious ownership" checks and
# pollutes the outer status.
GOLDEN_REPO_DIR="${GOLDEN_REPO_DIR:-$(dirname "$REPO_ROOT")/training_framework_golden_files}"
mkdir -p "$(dirname "$GOLDEN_REPO_DIR")"

# Allow cross-user access (e.g. root in a container operating on a user-owned
# clone), otherwise git aborts with "detected dubious ownership".
if ! git config --global --get-all safe.directory 2>/dev/null | grep -qx "$GOLDEN_REPO_DIR"; then
    git config --global --add safe.directory "$GOLDEN_REPO_DIR"
fi

echo "=== Fetching golden files repo ==="
if [[ -z "$GOLDEN_REPO_URL" ]]; then
    # No golden repo URL configured: the golden files must already be prepared
    # locally under GOLDEN_REPO_DIR by the user; use them as-is and skip the
    # clone/update entirely.
    echo "GOLDEN_REPO_URL is not set; using golden files under '${GOLDEN_REPO_DIR}' as-is."
elif [[ -d "${GOLDEN_REPO_DIR}/.git" ]]; then
    # The URL is always explicitly provided here (an empty URL takes the
    # local-only branch above), so point the existing clone's origin at it;
    # credentials embedded in the URL keep working.
    git -C "$GOLDEN_REPO_DIR" remote set-url origin "$GOLDEN_REPO_URL"
    git -C "$GOLDEN_REPO_DIR" fetch origin main
    git -C "$GOLDEN_REPO_DIR" reset --hard origin/main
else
    rm -rf "$GOLDEN_REPO_DIR"
    git clone --branch main "$GOLDEN_REPO_URL" "$GOLDEN_REPO_DIR"
fi

TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"
# Normalize to major.minor.patch (strips +git.../dev/a0 suffixes), e.g. 2.12.0 -> torch_2_12_0
TORCH_VERSION_DIR="$(python -c 'import re, torch; v = torch.__version__; m = re.match(r"[0-9]+\.[0-9]+\.[0-9]+", v); print("torch_" + m.group(0).replace(".", "_"))')"
GOLDEN_FILE="${GOLDEN_REPO_DIR}/torch/${TORCH_VERSION_DIR}/attention/${PLATFORM}/attention_microbenchmark_golden.json"
if [[ ! -f "$GOLDEN_FILE" ]]; then
    echo "ERROR: golden file not found: ${GOLDEN_FILE}" >&2
    echo "Please provide a golden result for torch ${TORCH_VERSION} on platform '${PLATFORM}'." >&2
    exit 1
fi
echo "Torch version: ${TORCH_VERSION} (${TORCH_VERSION_DIR})"
echo "Golden file: ${GOLDEN_FILE}"

# ---------------------------------------------------------------------------
# Step 3: Verify attention-gym is installed
# ---------------------------------------------------------------------------
echo "=== Checking attention-gym installation ==="
python -c "import attn_gym" 2>/dev/null || {
    echo "ERROR: attention-gym is not installed. Please install it first:" >&2
    echo "  pip install git+https://github.com/meta-pytorch/attention-gym.git@main" >&2
    exit 1
}
pip show triton

# ---------------------------------------------------------------------------
# Step 4: Run the benchmark NUM_RUNS times and aggregate (trimmed mean)
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}/benchmarks/transformer"

RESULT_FILES=()
for i in $(seq 1 "$NUM_RUNS"); do
    echo "=== Run ${i}/${NUM_RUNS}: Clearing Triton and Inductor caches ==="
    rm -rf ~/.triton/cache
    rm -rf "${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_$(whoami)}"

    echo "=== Run ${i}/${NUM_RUNS}: Running attention microbenchmark ==="
    OUT_FILE="${TEST_REPORTS_DIR}/attention_microbenchmark_current_run${i}.json"
    python score_mod.py --config configs/config_basic.yaml \
        --output-json-for-dashboard "$OUT_FILE"
    RESULT_FILES+=("$OUT_FILE")
done

echo "=== Aggregating ${NUM_RUNS} runs (trimmed mean: drop ${TRIM_COUNT} fastest + ${TRIM_COUNT} slowest) ==="
CURRENT_FILE="${TEST_REPORTS_DIR}/attention_microbenchmark_current_final.json"
python "${SCRIPT_DIR}/aggregate_results.py" \
    --method trimmed_mean \
    --trim "$TRIM_COUNT" \
    --output "$CURRENT_FILE" \
    "${RESULT_FILES[@]}"

# ---------------------------------------------------------------------------
# Step 5: Compare the current result against the golden and gate on threshold
# ---------------------------------------------------------------------------
echo "=== Comparing current result against golden (metrics matching '${METRIC_FILTER}') ==="
DIFF_FILE="${TEST_REPORTS_DIR}/attention_microbenchmark_current_vs_golden.json"
if python "${SCRIPT_DIR}/compare_results.py" \
    --metric-contains "$METRIC_FILTER" \
    --fail-above "$MAX_REL_DIFF_THRESHOLD" \
    --output "$DIFF_FILE" \
    "$GOLDEN_FILE" "$CURRENT_FILE"; then
    echo "=== BENCHMARK PASSED: max '${METRIC_FILTER}' diff within ${MAX_REL_DIFF_THRESHOLD_PCT}% ==="
    echo "=== Diff report: ${DIFF_FILE} ==="
    exit 0
else
    echo "=== BENCHMARK FAILED: max '${METRIC_FILTER}' diff exceeded ${MAX_REL_DIFF_THRESHOLD_PCT}% ===" >&2
    echo "=== Diff report: ${DIFF_FILE} ===" >&2
    exit 1
fi
