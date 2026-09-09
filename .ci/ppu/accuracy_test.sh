#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：CUDA inductor 精度测试入口。
# 由 .github/workflows/ppu-accuracy.yml 经 flytiger-eco/ppu-distributed-action
# 在单卡 PPU pod 内执行（源码已由 action 解压到 pod 的 source_dir）。
#
# 单独抽成脚本而不是内联到 yaml 的 command：command 由 pod 的默认 shell 执行，
# 未必是 bash，而 sdk_env.sh 依赖 bash 语法（[[ ]] / BASH_SOURCE）且必须被 source。
#
# 依赖环境变量：
#   PIP_INDEX        - 内部 pip 源（安装 torch 用，缺省用内部 pypiindex）
#   TORCH_VERSION    - 待安装的 torch 版本（默认 2.11.0）
#   SDK_INSTALL_DIR  - PPU SDK 安装目录（默认 /usr/local，供 sdk_env.sh 使用）
#   PR_NUMBER        - 仅用于日志溯源（可选）
# =============================================================================
set -euo pipefail

PIP_INDEX="${PIP_INDEX:-https://pkg.flytiger-eco.com/artifactory/api/pypi/pypiindex/simple}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

# 切到源码根目录：action 的 source_dir 可配置，这里按脚本自身位置反推，不写死路径
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[accuracy] 源码目录: $(pwd)"

# 跳过 build whl：直接装内部源预编译好的 torch
python -m pip install "torch==${TORCH_VERSION}" -i "$PIP_INDEX"

# sdk_env.sh 会 source PPU SDK 的 envsetup.sh（设置 LD_LIBRARY_PATH / PATH）并校验 nvcc。
# 必须 source（而非执行），环境变量才能作用于后续的测试进程。
source .ci/ppu/sdk_env.sh

echo "=== 环境自检 (CUDA) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "rank=${RANK:-0} nproc_per_node=${NPROC_PER_NODE:-1}"
python --version
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"

echo "=== CUDA inductor 精度单测（run_test.py --include 白名单过滤，仅 CUDA 相关） ==="
# 下面的 include 白名单是"可调项"：先给一组有代表性的 CUDA 精度用例，
# 请按 PPU 实际支持情况增删（例如想加深 op 级覆盖可加 inductor/test_torchinductor_opinfo，
# 但该文件很重、耗时长，默认不放进门禁）。
# 不使用 --upload-artifacts-while-running：那是官方 S3 上传路径，自建集群上没有。
python test/run_test.py \
    --include \
        inductor/test_torchinductor \
        inductor/test_torchinductor_dynamic_shapes \
        inductor/test_cuda_repro \
        inductor/test_cudagraph_trees \
        inductor/test_gpu_select_algorithm \
        inductor/test_fp8 \
    --verbose

echo "[accuracy] 完成"
