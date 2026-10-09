#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
source .ci/ppu/cuda_only_filter.sh

echo "=== CUDA inductor 精度单测（run_test.py --include 白名单过滤，仅 CUDA 相关） ==="

K_FP8="not fp8 and not FP8 and not Fp8 \
and not float8 and not Float8 \
and not e4m3 and not E4M3 \
and not e5m2 and not E5M2"

K_SKIP_CASES="not test_not_disabling_ftz_yields_zero \
and not test_triton_interpret \
and not test_graph_partition_user_defined_triton_kernel_reuse \
and not test_graph_partition_reorder_cpu_and_gpu_interleave \
and not test_avg_pool3d_backward2_cuda  \
and not test_consecutive_split_ \
and not test_linalg_eig_stride_consistency_cuda \
and not test_sort_stable_cuda \
and not test_copy_non_blocking_is_pinned_use_cat_True_cuda \
and not test_split_ \
and not RNN \
and not LSTM \
and not GRU \
and not test_put_cuda_float16 \
and not test_corrcoef_cuda_complex \
and not test_cov_cuda_complex"

echo "=== CUDA inductor 精度单测（单卡全量）==="
python test/run_test.py --inductor \
    --include \
        inductor/test_cuda_repro \
        inductor/test_cudagraph_trees \
        inductor/test_cuda_select_algorithm \
        inductor/test_torchinductor \
        test_modules \
        test_torch \
    -k "$(ppu_cuda_only_k_expr "$K_FP8" "$K_SKIP_CASES")" \
    --verbose

echo "[accuracy] 完成"
