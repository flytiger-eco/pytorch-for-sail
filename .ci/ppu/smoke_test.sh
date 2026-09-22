#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[smoke] 源码目录: $(pwd)"
source .ci/ppu/sdk_env.sh
bash .ci/ppu/install_wheel.sh

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
and (not test_triton_template_generated_code_caching or test_triton_template_generated_code_caching_bmm or test_triton_template_generated_code_caching_mm_plus_mm)"

python test/run_test.py \
    --include \
        test_matmul_cuda \
        test_scaled_matmul_cuda \
        nn/attention/test_open_registry \
        inductor/test_flex_flash \
        inductor/test_nv_universal_gemm \
        inductor/test_max_autotune \
    -k "$K_SKIP_CASES" \
    --verbose

echo "[smoke] 完成"
