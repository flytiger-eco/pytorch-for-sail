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
#   PIP_INDEX        - 内部 pip 源（可选；不设则用镜像自带的 pip 默认源）
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
# 自检必须在源码树外执行：`python -c` 会把 cwd（此时是源码根）放进 sys.path[0]，
# 未编译的源码目录 torch/ 会遮蔽镜像预装的 torch，而 torch/version.py 是构建产物、
# 源码树里不存在，于是报 ModuleNotFoundError: No module named 'torch.version'。
# 打印 torch.__file__ 以便一眼确认导入的是 site-packages 里的那份。
(cd /tmp && python -c "import torch; print('torch', torch.__version__, torch.__file__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())")
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"

echo "=== 安装测试框架依赖 ==="
# run_test.py 的 check_pip_packages() 硬校验这三个 pytest 插件，缺任意一个直接 exit 1，
# 镜像里没预装，所以必须在这里补。
# 注意：不照 run_test.py 报错提示执行 `pip install -r .ci/docker/requirements-ci.txt`。
# 那个文件钉了 numpy/sympy/onnx 等几十个版本，会重装、降级镜像里预装 torch 所依赖的包，
# 有把 PPU torch 环境搞坏的风险。这里只装 run_test.py 真正要求的测试框架。
# 版本与 .ci/docker/requirements-ci.txt 保持一致；不钉 pytest 自身版本，避免降级镜像自带的 pytest。
PIP_INSTALL=(python -m pip install --disable-pip-version-check)
if [[ -n "${PIP_INDEX:-}" ]]; then
    PIP_INSTALL+=(-i "${PIP_INDEX}")
fi
"${PIP_INSTALL[@]}" "pytest-rerunfailures>=10.3" "pytest-flakefinder==1.1.0" "pytest-xdist==3.3.1"

# boto3 故意不装：它只被 tools/stats/upload_metrics.py 用于往官方 S3 上报指标，缺失时
# EMIT_METRICS=False 静默降级（日志里那条 "Unable to import boto3" 只是提示，不影响退出码），
# 自建集群也没有对应凭证。

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
