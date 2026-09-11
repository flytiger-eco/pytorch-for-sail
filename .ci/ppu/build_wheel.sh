#!/usr/bin/env bash
# =============================================================================
# 容器内：源码编译 PPU torch whl。
# 方法来自 PPU fork README "Build from Source"（已人工验证可行）：
#   直接 python3 setup.py bdist_wheel，不经过 .ci/pytorch/build.sh。
# 依赖环境变量：
#   REPO_DIR                - pytorch 源码路径（必填）
#   BUILD_ENVIRONMENT       - 环境字符串（仅用于日志/产物标识）
#   TORCH_CUDA_ARCH_LIST    - 编译目标架构（默认 8.0；PPU 语义下 8.0 = SM80+SM89 混合编译）
#   MAX_JOBS                - 编译并行度（0 或空 = nproc-2）
#   PIP_INDEX_URL           - pip 源（可选）
#   BUILD_ADDITIONAL_PACKAGES - 保留参数；README 直编方式不处理额外包，仅告警
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR 未设置}"
BUILD_ENVIRONMENT="${BUILD_ENVIRONMENT:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
BUILD_ADDITIONAL_PACKAGES="${BUILD_ADDITIONAL_PACKAGES:-}"
MAX_JOBS="${MAX_JOBS:-0}"

# ---- 复用已有产物（同 commit 同架构，stamp 存在则跳过编译） ----
# FORCE_REBUILD=1 强制重编
# stamp 以"源码基线 commit"为键（fetch_repo 写入；yaml 注入产生的本地提交不影响编译产物），
# 避免 HEAD 因 yaml 重新注入漂移导致跨 Run 复用失效
COMMIT=$(cat /tmp/ppu_source_commit 2>/dev/null || git -C "$REPO_DIR" rev-parse --short HEAD)
WHL_STAMP="$REPO_DIR/dist/.built_${COMMIT}_${TORCH_CUDA_ARCH_LIST// /_}"
if [[ -f "$WHL_STAMP" && "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[build_wheel] 检测到已有编译产物（stamp $WHL_STAMP），跳过编译"
    ls "$REPO_DIR"/dist/torch-*.whl
    exit 0
fi

# SDK 前置（需求 9）
source "$(dirname "$0")/sdk_env.sh"

# 编译缓存前置：装好 ccache、写配置、导出 CCACHE_DIR 与 CMAKE_*_COMPILER_LAUNCHER。
# 缓存目录由宿主机挂载进来（默认 /root/.cache/ccache），跨 run / 跨 PR 复用由
# workflow 侧的 actions/cache 负责；ccache 装不上时只告警，不阻断编译。
source "$(dirname "$0")/ccache_env.sh"

cd "$REPO_DIR"
echo "[build_wheel] BUILD_ENVIRONMENT=$BUILD_ENVIRONMENT"
echo "[build_wheel] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

if [[ -n "$BUILD_ADDITIONAL_PACKAGES" ]]; then
    echo "[build_wheel] 警告: README 直编方式不支持 build-additional-packages: $BUILD_ADDITIONAL_PACKAGES（忽略）" >&2
fi

# 固化 pip 源
if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    mkdir -p /root/.config/pip
    printf '[global]\nindex-url = %s\n' "$PIP_INDEX_URL" > /root/.config/pip/pip.conf
fi

# ---- 修复镜像自带工具链：cmake 版本过高、pybind11 与 py3.12 不兼容 ----
apt remove -y cmake >/dev/null 2>&1 || true
pip uninstall -y cmake >/dev/null 2>&1 || true
pip cache purge >/dev/null 2>&1 || true
pip install cmake==3.31.10
ln -sf /usr/local/bin/cmake /usr/bin/cmake
pip install --upgrade pybind11
echo "[build_wheel] cmake=$(cmake --version | head -1)"

# ---- 安装编译依赖（README step 2）----
pip install -r requirements.txt

# ---- 并行度 ----
# 除了听 MAX_JOBS，还要在脚本里兜一层内存上限：
#   8vCPU/32GB 的 runner 上把并行度拉到 2×核数时，nvcc 背后的 cicc 单个 TU 峰值
#   可达 3~5GB，十几个并发直接把 32GB 打穿，内核 OOM killer 上场；对应现象是
#   日志里出现 Killed、docker exec 返回 137。
# PPU_MEM_PER_JOB_GB 默认 4：32GB → 最多 8 并发。
if [[ -z "$MAX_JOBS" || "$MAX_JOBS" == "0" ]]; then
    MAX_JOBS=$(nproc)
    MAX_JOBS=$((MAX_JOBS > 2 ? MAX_JOBS - 2 : 1))
fi

# 读 /proc/meminfo 而不用 free：后者依赖 procps，镜像里不一定装了
MEM_GB=$(awk '/^MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo 2>/dev/null || echo 0)
MEM_PER_JOB_GB="${PPU_MEM_PER_JOB_GB:-4}"
if [[ "${MEM_GB:-0}" -gt 0 ]]; then
    MEM_CAP=$((MEM_GB / MEM_PER_JOB_GB))
    if [[ "$MEM_CAP" -lt 1 ]]; then
        MEM_CAP=1
    fi
    if [[ "$MAX_JOBS" -gt "$MEM_CAP" ]]; then
        echo "[build_wheel] MAX_JOBS=${MAX_JOBS} 超过内存上限（${MEM_GB}GB / ${MEM_PER_JOB_GB}GB per job），下调到 ${MEM_CAP}"
        MAX_JOBS="$MEM_CAP"
    fi
fi
export MAX_JOBS

# 链接阶段单独限流：ld 链 libtorch_cuda.so（两个架构的 fatbin）以及 BUILD_TEST 那
# 一堆测试二进制时，单个 ld 就能吃掉数 GB，ninja 默认会把它们与编译一起并发，
# 而且链接集中在构建末尾——之前 92% 处被掉断就发生在这一段。
# 用 ninja job pool 把链接并发压到 PPU_LINK_JOBS（默认 2），编译仍按 MAX_JOBS 跑。
# CMAKE_ 前缀的环境变量会被 setup.py 自动透传成 -D 交给 cmake（tools/setup_helpers/cmake.py）。
PPU_LINK_JOBS="${PPU_LINK_JOBS:-2}"
export CMAKE_JOB_POOLS="compile=${MAX_JOBS};link=${PPU_LINK_JOBS}"
export CMAKE_JOB_POOL_COMPILE=compile
export CMAKE_JOB_POOL_LINK=link

# 版本号取 version.txt 的 x.y.z 段（兼容 2.11.0 / 2.11.0a0+gitxxx）
TORCH_VERSION=$(sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/' version.txt)
echo "[build_wheel] TORCH_VERSION=${TORCH_VERSION} MAX_JOBS=${MAX_JOBS} 链接并发=${PPU_LINK_JOBS} 内存=${MEM_GB}GB"

BUILD_START=$(date +%s)

# ---- README step 3：直接编译 ----
env \
    NCCL_INCLUDE_DIR="$SDK_INSTALL_DIR/PPU_SDK/CUDA_SDK/include" \
    NCCL_LIB_DIR="$SDK_INSTALL_DIR/PPU_SDK/CUDA_SDK/lib64" \
    PYTORCH_VERSION="$TORCH_VERSION" \
    PYTORCH_BUILD_VERSION="$TORCH_VERSION" \
    PYTORCH_BUILD_NUMBER=0 \
    USE_FLASH_ATTENTION=True \
    USE_MEM_EFF_ATTENTION=True \
    USE_NCCL=True \
    USE_DISTRIBUTED=True \
    USE_SYSTEM_NCCL=1 \
    BUILD_CAFFE2=False \
    BUILD_TEST=True \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    python3 setup.py bdist_wheel

BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))
echo "$BUILD_DURATION" > /tmp/ppu_build_duration_sec
echo "[build_wheel] 编译耗时 ${BUILD_DURATION}s"

# 产物校验
WHLS=("$REPO_DIR/dist"/torch-*.whl)
if [[ ! -e "${WHLS[0]}" ]]; then
    echo "[build_wheel] 错误: 未找到 dist/torch-*.whl" >&2
    exit 1
fi
echo "[build_wheel] 产物: $(ls "$REPO_DIR/dist"/torch-*.whl)"

# 本轮命中情况（workflow 侧另有一份写进 job summary）
if [[ "${PPU_CCACHE_ENABLED:-0}" == "1" ]]; then
    ccache -s 2>/dev/null | sed 's/^/[build_wheel] ccache /' || true
fi

# 写入 stamp 供下次复用
echo "${WHLS[*]##*/}" > "$WHL_STAMP"

echo "[build_wheel] 完成"
