#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：Inductor 精度测试入口。
# 由 .github/workflows/ppu-accuracy.yml 经 flytiger-eco/ppu-distributed-action
# 在单卡 PPU pod 内执行（源码已由 action 解压到 pod 的 source_dir）。
#
# 单独抽成脚本而不是内联到 yaml 的 command：command 由 pod 的默认 shell 执行，
# 未必是 bash，而 sdk_env.sh 依赖 bash 语法（[[ ]] / BASH_SOURCE）且必须被 source。
#
# 用法：
#   bash accuracy_test.sh cuda   # PPU 设备可见，跑 CUDA 相关 inductor 用例
#   bash accuracy_test.sh cpu    # 屏蔽设备，强制 inductor 走纯 CPU 代码路径
#
# 依赖环境变量：
#   PIP_INDEX        - 内部 pip 源（安装 torch 用，缺省用内部 pypiindex）
#   TORCH_VERSION    - 待安装的 torch 版本（默认 2.11.0）
#   SDK_INSTALL_DIR  - PPU SDK 安装目录（默认 /usr/local，供 sdk_env.sh 使用）
#   PR_NUMBER        - 仅用于日志溯源（可选）
# =============================================================================
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "cuda" && "$MODE" != "cpu" ]]; then
    echo "[accuracy] 用法: $0 <cuda|cpu>" >&2
    exit 1
fi

PIP_INDEX="${PIP_INDEX:-https://pkg.flytiger-eco.com/artifactory/api/pypi/pypiindex/simple}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

# 切到源码根目录：action 的 source_dir 可配置，这里按脚本自身位置反推，不写死路径
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[accuracy] mode=$MODE 源码目录: $(pwd)"

# 跳过 build whl：直接装内部源预编译好的 torch
python -m pip install "torch==${TORCH_VERSION}" -i "$PIP_INDEX"

# sdk_env.sh 会 source PPU SDK 的 envsetup.sh（设置 LD_LIBRARY_PATH / PATH）并校验 nvcc。
# 必须 source（而非执行），环境变量才能作用于后续的测试进程。
# CPU 模式同样需要：PPU 版 torch 依赖 SDK 的 LD_LIBRARY_PATH 才能 import 成功。
source .ci/ppu/sdk_env.sh

echo "=== 环境自检 ($MODE) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "rank=${RANK:-0} nproc_per_node=${NPROC_PER_NODE:-1}"
python --version

if [[ "$MODE" == "cuda" ]]; then
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
else
    # 纯 CPU：屏蔽 CUDA/PPU 设备，强制 inductor 走 CPU 代码路径（CUDA 用例会自动 skip）。
    # 若 PPU SDK 用的是别的设备可见性变量（而非 CUDA_VISIBLE_DEVICES），请在此一并置空。
    export CUDA_VISIBLE_DEVICES=""

    python -c "import torch; print('torch', torch.__version__); print('cuda_available(expect False)', torch.cuda.is_available())"

    echo "=== 纯 CPU inductor 精度单测（run_test.py --include 白名单过滤，仅 CPU 相关） ==="
    # 同样是"可调项"：一组代表性的 CPU 精度/代码生成用例，按 PPU CPU 后端支持情况增删。
    python test/run_test.py \
        --include \
            inductor/test_cpu_repro \
            inductor/test_cpu_select_algorithm \
            inductor/test_loop_ordering \
            inductor/test_mix_order_reduction \
            inductor/test_inductor_freezing \
            inductor/test_mkldnn_pattern_matcher \
        --verbose
fi

echo "[accuracy] mode=$MODE 完成"
