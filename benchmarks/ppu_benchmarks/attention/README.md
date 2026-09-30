# PPU Attention Benchmark (Golden Comparison)

This directory contains the PPU attention benchmark harness. The entry point
is `run.sh`, which runs the attention microbenchmark on the current machine
and gates the result against a stored golden baseline.

## What run.sh does

1. **Platform detection** — reads the CUDA device name via
   `torch.cuda.get_device_name(0)` and maps it to a platform tag (`M890`,
   `810E`, or `H200`). Unrecognized devices abort the run.
2. **Golden file preparation** — clones (or updates, via `fetch` + `reset
   --hard`) a user-provided `training_framework_golden_files` repository
   next to the pytorch repo (outside its worktree) when `GOLDEN_REPO_URL` is
   set, or falls back to a locally prepared `GOLDEN_REPO_DIR` when it is not,
   then selects the golden file by the installed torch version and the
   detected platform:

   ```text
   training_framework_golden_files/torch/torch_<X_Y_Z>/attention/<PLATFORM>/attention_microbenchmark_golden.json
   ```

   The torch version is normalized to `major.minor.patch` (e.g.
   `2.12.0.dev20260814+git...` → `torch_2_12_0`).
3. **Dependency check** — verifies that `attention-gym` (`attn_gym`) is
   importable; prints install instructions otherwise.
4. **Benchmark execution** — runs
   `benchmarks/transformer/score_mod.py --config configs/config_basic.yaml`
   three times, clearing the Triton and Inductor caches before each run, and
   aggregates the runs into a single result using the median
   (`aggregate_results.py`).
5. **Golden comparison** — compares the aggregated result against the golden
   file (`compare_results.py`), judging only metrics whose names contain
   `latency`. The run **passes** if the maximum absolute relative difference
   is within 5%, and **fails** (exit code 1) otherwise.

## Exit codes

| Exit code | Meaning                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------ |
| `0`       | Benchmark passed: the maximum `latency` relative difference against the golden is within 5%.    |
| `1`       | Benchmark failed: the maximum `latency` relative difference exceeded 5%, or a setup error occurred (unrecognized platform, golden file missing for this torch version/platform, `attention-gym` not installed). Any command failing mid-run also aborts with a non-zero exit code due to `set -euo pipefail`. |

The final verdict line is printed at the end: `BENCHMARK PASSED ...` or
`BENCHMARK FAILED ...`, and the per-metric diff report is always written to
`<pytorch_repo>/test/test-reports/attention_microbenchmark_current_vs_golden.json`.

## Usage

```bash
cd pytorch/benchmarks/ppu_benchmarks/attention
./run.sh
```

The golden baseline files are **not** distributed with this repository: they
live in a separate `training_framework_golden_files` repository that each user
must prepare on their own (for example, a self-hosted git repository, or a
local copy of the golden files). Point the harness at your golden files with:

```bash
# Option A: a git repository you can access (cloned/updated on each run)
export GOLDEN_REPO_URL="https://<your-git-host>/<you>/training_framework_golden_files.git"

# Option B: a local directory that already holds the golden files
export GOLDEN_REPO_DIR="/path/to/training_framework_golden_files"
```

When `GOLDEN_REPO_URL` is not set, `run.sh` skips the clone/update step and
uses the contents of `GOLDEN_REPO_DIR` as-is.

### Environment variables

| Variable              | Default                                              | Purpose                                        |
| --------------------- | ---------------------------------------------------- | ---------------------------------------------- |
| `GOLDEN_REPO_URL`     | *(empty)*                                            | Optional URL of a user-provided git repository holding the golden files; when empty, the clone/update step is skipped and `GOLDEN_REPO_DIR` is used as-is |
| `GOLDEN_REPO_DIR`     | `<pytorch_repo_parent>/training_framework_golden_files` | Local location of the golden files       |
| `TORCHINDUCTOR_CACHE_DIR` | `/tmp/torchinductor_$(whoami)`                  | Inductor cache path cleared before each run    |

## Outputs

All reports are written to `<pytorch_repo>/test/test-reports/`:

- `attention_microbenchmark_current_run{1..3}.json` — per-run raw results
- `attention_microbenchmark_current_final.json` — median-aggregated result
- `attention_microbenchmark_current_vs_golden.json` — per-metric diff against
  the golden

## Helper scripts

- `aggregate_results.py` — aggregates multiple benchmark result JSON files
  (e.g. median across runs).
- `compare_results.py` — compares two result JSON files metric by metric,
  with optional metric-name filtering and a relative-difference gate.

## Adding a new golden

Golden files live in a user-provided `training_framework_golden_files`
repository under
`torch/torch_<X_Y_Z>/attention/<PLATFORM>/attention_microbenchmark_golden.json`.
To add coverage for a new torch version or platform, generate the JSON with
the same benchmark command and commit it to your golden files repository.
