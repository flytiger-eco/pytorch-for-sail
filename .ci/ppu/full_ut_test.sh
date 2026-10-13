#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

if [[ -z "${SHARD_NUMBER:-}" || -z "${NUM_TEST_SHARDS:-}" ]]; then
    echo "[full-ut][error] 必须传 SHARD_NUMBER 与 NUM_TEST_SHARDS（由 workflow 的 matrix 给）" >&2
    exit 1
fi
if (( NUM_TEST_SHARDS != 2 )); then
    echo "[full-ut][error] 本脚本按 2 分片手工编排，NUM_TEST_SHARDS 必须为 2，实际: ${NUM_TEST_SHARDS}" >&2
    exit 1
fi
if (( SHARD_NUMBER < 1 || SHARD_NUMBER > 2 )); then
    echo "[full-ut][error] SHARD_NUMBER 只能是 1 或 2，实际: ${SHARD_NUMBER}" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[full-ut] 源码目录: $(pwd)"
echo "[full-ut] 分片=${SHARD_NUMBER}/${NUM_TEST_SHARDS}（单卡 pod）"
source .ci/ppu/sdk_env.sh
source .ci/ppu/nccl_env.sh
bash .ci/ppu/install_wheel.sh

bash .ci/ppu/install_triton.sh

echo "=== 环境自检 (CUDA) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "rank=${RANK:-0} nproc_per_node=${NPROC_PER_NODE:-1}"
python --version
(cd /tmp && python -c "import torch; print('torch', torch.__version__, torch.__file__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())")
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"

bash .ci/ppu/install_test_deps.sh

FAILED_CASES=()
run_case() {
    local test_name="$1"
    local k_expr="${2:-}"
    local test_file="test/${test_name}.py"
    echo "=== [case] ${test_name}${k_expr:+  -k \"${k_expr}\"} ==="
    if [[ -n "$k_expr" ]]; then
        if python .ci/ppu/multi_run_case.py "$test_file" -k "$k_expr"; then
            echo "=== [pass] ${test_name} ==="
        else
            echo "::error::[fail] ${test_name}"
            FAILED_CASES+=("${test_name}")
        fi
    else
        if python .ci/ppu/multi_run_case.py "$test_file"; then
            echo "=== [pass] ${test_name} ==="
        else
            echo "::error::[fail] ${test_name}"
            FAILED_CASES+=("${test_name}")
        fi
    fi
}

shard_heavy() {
    run_case inductor/test_aot_inductor \
        "cuda and not test_aoti_load_package_in_fresh_subprocess_cuda and not test_repeated_calling_cuda and not test_simple_multi_arch_embed_kernel_binary_False_cuda"

    run_case test_sparse \
        "cuda and not test_mm_cuda_complex and not test_sparse_addmm and not test_sparse_matmul_cuda_bfloat16 and not test_sparse_matmul_cuda_complex and not test_sparse_matmul_cuda_float16 and not test_constructor_autograd_SparseBS and not test_gradcheck_mm_SparseCOO and not test_gradcheck_mm_SparseCSC and not test_gradcheck_mm_SparseCSR"
}

# shard 2：其余全部文件
shard_rest() {
    run_case test_torch \
        "not test_corrcoef_cuda_complex64 and not test_terminate_handler_on_crash and not test_cov_cuda_complex and not test_put_cuda_float16"

    run_case test_autograd "not test_thread_shutdown"

    run_case test_binary_ufuncs "TestBinaryUfuncsCUDA"

    run_case test_unary_ufuncs "not TestUnaryUfuncsCPU"

    run_case test_reductions "cuda and not test_ref_extremal_values_hash_tensor_cuda_float32"

    run_case test_shape_ops "TestShapeOpsCUDA"

    run_case test_indexing "cuda"

    run_case test_tensor_creation_ops "cuda"

    run_case test_dataloader \
        "not test_fd_limit_exceeded and not test_multiprocessing_contexts and not test_sparse_tensor_multiprocessing and not test_early_exit and not test_segfault"

    run_case test_optim "cuda"

    run_case test_serialization "cuda"

    run_case test_fx "cuda"

    run_case test_autocast "cuda"

    run_case test_type_promotion "cuda"

    run_case test_complex "cuda"

    run_case test_foreach "cuda"

    run_case test_sort_and_select "cuda"

    run_case nn/test_convolution \
        "cuda and not test_conv_transposed_large_cuda and not test_cudnn_convolution_add_relu_cuda and not test_cudnn_convolution_relu_cuda and not test_depthwise_conv_64bit_indexing_cuda"

    run_case nn/test_embedding

    run_case nn/test_pooling

    run_case dynamo/test_misc \
        "not test_outside_linear_module_free and not test_packaging_version_parse and not test_parameter_free and not test_pytree_tree_ and not test_tracing_nested and not test_tracing_pytree_cxx"

    run_case dynamo/test_compile

    run_case dynamo/test_repros

    run_case dynamo/test_export

    run_case distributed/test_c10d_common

    run_case test_dynamic_shapes "not test_do_not_guard_unbacked_inputs"
}

if (( SHARD_NUMBER == 1 )); then
    echo "=== CUDA 全量 UT（shard 1/2：重型三件套） ==="
    shard_heavy
else
    echo "=== CUDA 全量 UT（shard 2/2：其余全部文件） ==="
    shard_rest
fi

if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo "=== 失败清单（${#FAILED_CASES[@]} 个文件）==="
    for case_label in "${FAILED_CASES[@]}"; do
        echo "  - ${case_label}"
    done
    echo "::error::[full-ut] ${#FAILED_CASES[@]} 个测试文件失败，详见上方各 [case] 段落的日志"
    exit 1
fi

echo "[full-ut] 完成，全部通过（shard=${SHARD_NUMBER}/${NUM_TEST_SHARDS}）"
