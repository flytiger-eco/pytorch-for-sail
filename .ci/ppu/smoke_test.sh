#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[smoke] 源码目录: $(pwd)"
source .ci/ppu/sdk_env.sh
source .ci/ppu/nccl_env.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh

echo "=== 环境自检 ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "rank=${RANK:-0} nproc_per_node=${NPROC_PER_NODE:-1}"
python --version
(cd /tmp && python -c "import torch; print('torch', torch.__version__, torch.__file__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())")
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"
bash .ci/ppu/install_test_deps.sh
echo "=== CUDA 冒烟用例（run_test.py --include 精确过滤，仅 CUDA 相关） ==="

K_SKIP_CASES="not test_benchmark_choice_fail_in_subproc \
and not test_lazy_template_fusion_multiple_candidates_use_async_compile \
and not test_template_epilogue_fusion_static_analysis_test_case_spills_reject_use_async_compile \
and not test_template_epilogue_fusion_static_analysis_test_case_timing_reject_use_async_compile \
and not test_async_autotuner_cache_same_inputs \
and not test_bmm_out_dtype \
and not test_cat_max_autotune_extern \
and not test_compilation_after_inactivity \
and not test_linear_and_cel \
and not test_max_autotune_mm_plus_mm_zero_size_input_dynamic_False_search_space \
and not test_max_autotune_regular_mm_zero_size_input_dynamic \
and not test_mutation_rename \
and not test_cublas_baddbmm_large_input \
and not test_mm_bmm_dtype_overload_float16_M \
and not test_mm_with_mH_args_backend_cublas_cuda \
and not test_matmul_dropout_device_cpu \
and not test_float8_error_messages_cuda \
and not test_float8_rowwise_scaling_sanity_use_fast_accum \
and not test_float8_scale_fast_accum_cuda \
and not test_scaled_mm_vs_emulated_row_wise_bfloat16_shapes0_cuda \
and not test_float8_basics_cuda \
and not test_cublas_addmm_reduced_precision_fp16 \
and not test_mm_with_mH_args_backend_cublaslt_cuda \
and not test_template_epilogue_fusion_extra_reads_fuse_epilogue \
and not test_cublas_and_lt_reduced_precision_fp16_accumulate_cuda \
and not test_mixed_dtypes_linear_cuda_float16 \
and not test_fake_crossref_backward_no_amp_nn_functional_bilinear_cuda_float32 \
and not test_numpy_ref_nn_functional_pdist_cuda_float64 \
and not test_noncontiguous_samples_nn_functional_grid_sample_cuda_float32 \
and not test_triton_template_generated_code_caching \
and not test_triton_template_generated_code_caching_bmm \
and not test_triton_template_generated_code_caching_mm_plus_mm"

PASSED_CASES=()
FAILED_CASES=()
run_case() {
    local test_name="$1"
    echo "=== [case] ${test_name} ==="
    if python test/run_test.py \
        --include "${test_name}" \
        -k "$K_SKIP_CASES" \
        --verbose; then
        echo "=== [pass] ${test_name} ==="
        PASSED_CASES+=("${test_name}")
    else
        echo "::error::[fail] ${test_name}"
        FAILED_CASES+=("${test_name}")
    fi
}

run_case test_matmul_cuda
run_case test_scaled_matmul_cuda
run_case nn/attention/test_open_registry
run_case inductor/test_flex_flash
run_case inductor/test_nv_universal_gemm
run_case inductor/test_max_autotune

total=$(( ${#PASSED_CASES[@]} + ${#FAILED_CASES[@]} ))
echo "=== 冒烟测试汇总（共 ${total} 个文件：通过 ${#PASSED_CASES[@]}，失败 ${#FAILED_CASES[@]}）==="
if [ "${#PASSED_CASES[@]}" -ne 0 ]; then
    for case_label in "${PASSED_CASES[@]}"; do
        echo "  [pass] ${case_label}"
    done
fi
if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    for case_label in "${FAILED_CASES[@]}"; do
        echo "  [fail] ${case_label}"
    done
    echo "::error::[smoke] ${#FAILED_CASES[@]} 个测试文件失败，详见上方各 [case] 段落的日志"
    exit 1
fi

echo "[smoke] 完成，全部通过"
