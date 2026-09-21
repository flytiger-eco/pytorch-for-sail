#!/usr/bin/env bash
set -uo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[distributed] 源码目录: $(pwd)"

source .ci/ppu/sdk_env.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh

echo "=== 环境自检 (CUDA, 2 卡) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "action 注入: NNODES=${NNODES:-} NPROC_PER_NODE=${NPROC_PER_NODE:-} RANK=${RANK:-}" \
     "LOCAL_RANK=${LOCAL_RANK:-} WORLD_SIZE=${WORLD_SIZE:-} NODE_RANK=${NODE_RANK:-}" \
     "MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-}"
unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK MASTER_ADDR MASTER_PORT
echo "已清理上述 rank / rendezvous 变量，交由各用例自行 spawn 子进程"

python --version
(cd /tmp && python - <<'PY'
import sys

import torch

print("torch", torch.__version__, torch.__file__)
print("cuda_available", torch.cuda.is_available())
count = torch.cuda.device_count()
print("device_count", count)
if count < 2:
    sys.exit(
        f"[distributed] 需要至少 2 张 PPU，实际只看到 {count} 张。"
        "本门禁全部用例都是多卡语义，卡数不足时它们只会被 skip（假通过），故直接失败。"
        "请检查 workflow 的 nproc_per_node 是否为 2、以及 pod 是否真的拿到了 2 张卡。"
    )
PY
)
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"
bash .ci/ppu/install_test_deps.sh
source .ci/ppu/cuda_only_filter.sh

K_FP8_EXPR="not fp8 and not FP8 and not Fp8 \
and not float8 and not Float8 \
and not e4m3 and not E4M3 \
and not e5m2 and not E5M2 \
and not scaled_matmul \
and not test_fixed_striding"

K_SKIP_CASES="not test_dtensor_seq_par_shard_dim_0 \
and not test_set_reduce_scatter_divide_factor \
and not test_basic_all_gather_bucketing \
and not test_schedule_overlap_benchmark \
and not test_bucket_exposed_with_hidden_single_overlap \
and not test_bucketing_split_for_overlap \
and not test_bucketing_wait_sink \
and not test_fully_shard_force_sum_reduce_scatter"

K_FP8="$(ppu_cuda_only_k_expr "$K_FP8_EXPR" "$K_SKIP_CASES")"
FAILED_CASES=()
run_case() {
    local own_k="$1"
    shift
    local k_expr label
    if [ -n "$own_k" ]; then
        k_expr="( ${own_k} ) and ( ${K_FP8} )"
    else
        k_expr="${K_FP8}"
    fi
    label="$*${own_k:+ -k ${own_k}}"

    echo "=== [case] ${label} ==="
    if python test/run_test.py --include "$@" -k "$k_expr" --verbose; then
        echo "=== [pass] ${label} ==="
    else
        echo "::error::[fail] ${label}"
        FAILED_CASES+=("${label}")
    fi
}

echo "=== CUDA 分布式用例（2 卡；仅 CUDA、已排除 FP8） ==="

run_case "" distributed/test_c10d_functional_native
run_case "" distributed/tensor/test_dtensor_compile
run_case "" distributed/tensor/parallel/test_micro_pipeline_tp
run_case "" distributed/_composable/test_replicate_with_compiler

run_case "test_train_parity_multi_group"                   distributed/_composable/fsdp/test_fully_shard_training
run_case "test_train_parity_with_activation_checkpointing" distributed/_composable/fsdp/test_fully_shard_training
run_case "test_train_parity_hsdp"                          distributed/_composable/fsdp/test_fully_shard_training
run_case "test_gradient_accumulation"                      distributed/_composable/fsdp/test_fully_shard_training
run_case "test_dp_state_dict_save_load" distributed/_composable/fsdp/test_fully_shard_state_dict
run_case "test_compute_dtype" distributed/_composable/fsdp/test_fully_shard_mixed_precision
run_case "test_reduce_dtype"  distributed/_composable/fsdp/test_fully_shard_mixed_precision
run_case "test_clip_grad_norm" distributed/_composable/fsdp/test_fully_shard_clip_grad_norm_

run_case "" \
    distributed/test_inductor_collectives \
    distributed/test_aten_comm_compute_reordering \
    distributed/test_compute_comm_reordering

if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo "=== 失败清单（${#FAILED_CASES[@]} 条）==="
    for case_label in "${FAILED_CASES[@]}"; do
        echo "  - ${case_label}"
    done
    echo "::error::[distributed] ${#FAILED_CASES[@]} 条用例失败，详见上方各 [case] 段落的日志"
    exit 1
fi

echo "[distributed] 完成，全部通过"
