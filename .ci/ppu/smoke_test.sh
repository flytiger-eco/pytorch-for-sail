#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：CUDA 冒烟测试入口。
# 由 .github/workflows/ppu_smoke.yml 经 flytiger-eco/ppu-distributed-action
# 在单卡 PPU pod 内执行（源码已由 action 解压到 pod 的 source_dir）。
#
# 单独抽成脚本而不是内联到 yaml 的 command：command 由 pod 的默认 shell 执行，
# 未必是 bash，而 sdk_env.sh 依赖 bash 语法（[[ ]] / BASH_SOURCE）且必须被 source。
#
# torch 由 PPU pytorch 发布镜像预装，本脚本不再 pip install torch。
#
# 依赖环境变量：
#   SDK_INSTALL_DIR  - PPU SDK 安装目录（默认 /usr/local，供 sdk_env.sh 使用）
#   PR_NUMBER        - 仅用于日志溯源（可选）
# =============================================================================
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

# 切到源码根目录：action 的 source_dir 可配置，这里按脚本自身位置反推，不写死路径
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[smoke] 源码目录: $(pwd)"

# sdk_env.sh 会 source PPU SDK 的 envsetup.sh（设置 LD_LIBRARY_PATH / PATH）并校验 nvcc。
# 必须 source（而非执行），环境变量才能作用于后续的测试进程。
source .ci/ppu/sdk_env.sh

echo "=== 环境自检 ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "rank=${RANK:-0} nproc_per_node=${NPROC_PER_NODE:-1}"
python --version
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"

echo "=== CUDA 冒烟用例（run_test.py --include 精确过滤，仅 CUDA 相关） ==="
# 对齐 .ci/pytorch/test.sh 的 test_python_smoke，裁剪为 PPU 单卡可跑的子集：
#   去掉 inductor/test_max_autotune、inductor/test_cutedsl_grouped_mm、
#   inductor/test_flex_attention(-k test_tma_with_customer_kernel_options) 等
#   H100/B200(SM90/TMA/CUTLASS) 专属用例，它们在真武 PPU 上不适用。
# 不使用 --upload-artifacts-while-running：那是官方 S3 上传路径，自建集群上没有。
python test/run_test.py \
    --include \
        test_matmul_cuda \
        test_scaled_matmul_cuda \
        inductor/test_fp8 \
    --verbose

echo "[smoke] 完成"
