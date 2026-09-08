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

# 编译缓存持久化（宿主机挂载）；存在才启用
if [[ -d /root/.cache/ccache ]]; then
    export CCACHE_DIR=/root/.cache/ccache
    echo "[build_wheel] ccache 启用: $CCACHE_DIR ($(ccache --show-stats 2>/dev/null | grep 'Cache size' || echo '新建'))"
fi

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
if [[ -z "$MAX_JOBS" || "$MAX_JOBS" == "0" ]]; then
    MAX_JOBS=$(nproc)
    MAX_JOBS=$((MAX_JOBS > 2 ? MAX_JOBS - 2 : 1))
fi
export MAX_JOBS

# 版本号取 version.txt 的 x.y.z 段（兼容 2.11.0 / 2.11.0a0+gitxxx）
TORCH_VERSION=$(sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/' version.txt)
echo "[build_wheel] TORCH_VERSION=$TORCH_VERSION MAX_JOBS=$MAX_JOBS"

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

# 写入 stamp 供下次复用
echo "${WHLS[*]##*/}" > "$WHL_STAMP"

echo "[build_wheel] 完成"
