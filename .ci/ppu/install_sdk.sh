#!/usr/bin/env bash
# =============================================================================
# 容器内：下载并安装 PPU SDK（含定制 CUDA），必须在安装 torch whl 之前执行
# 安装方式与人工操作一致：
#   wget <SDK_URL> && tar -zxf ... -C /usr/local/
#   ln -s /usr/local/PPU_SDK/CUDA_SDK /usr/local/cuda
#   source /usr/local/PPU_SDK/envsetup.sh
# =============================================================================
set -euo pipefail

# 兜底默认地址（正常由 runner.py 按 config.py 的 sdk.url 注入 SDK_URL，改动需与其同步）
DEFAULT_SDK_URL="https://pkg.flytiger-eco.com/artifactory/generic-local/CUDA_SDK/v2.1.1/PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e.tar.gz"
SDK_URL="${SDK_URL:-$DEFAULT_SDK_URL}"
SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
INSTALLED_MARKER="$SDK_INSTALL_DIR/PPU_SDK/.ci_installed"

# ---- 幂等：已安装且标记存在则只做自检 ----
if [[ -f "$INSTALLED_MARKER" ]]; then
    echo "[install_sdk] 检测到已安装 SDK，跳过下载: $SDK_INSTALL_DIR/PPU_SDK"
else
    TARBALL="/tmp/$(basename "$SDK_URL")"
    echo "[install_sdk] 下载: $SDK_URL"
    wget -q --show-progress -O "$TARBALL" "$SDK_URL"
    echo "[install_sdk] 解压到: $SDK_INSTALL_DIR"
    tar -zxf "$TARBALL" -C "$SDK_INSTALL_DIR"
    rm -f "$TARBALL"
    [[ -d "$SDK_INSTALL_DIR/PPU_SDK" ]] || {
        echo "[install_sdk] 解压后未找到 $SDK_INSTALL_DIR/PPU_SDK" >&2; exit 1; }
fi

# ---- 软链 CUDA_SDK -> /usr/local/cuda（已存在则跳过） ----
if [[ ! -e /usr/local/cuda ]]; then
    ln -s "$SDK_INSTALL_DIR/PPU_SDK/CUDA_SDK" /usr/local/cuda
    echo "[install_sdk] 已创建软链: /usr/local/cuda -> PPU_SDK/CUDA_SDK"
else
    echo "[install_sdk] /usr/local/cuda 已存在，跳过软链"
fi

# ---- 自检：envsetup.sh 生效 + nvcc 可用 ----
set +u
source "$SDK_INSTALL_DIR/PPU_SDK/envsetup.sh"
set -u
echo "[install_sdk] envsetup.sh 已 source"
command -v nvcc && nvcc --version | tail -2 || echo "[install_sdk] 警告: nvcc 不在 PATH" >&2

# ---- 落标记，供 sdk_env.sh 快速判断 ----
touch "$INSTALLED_MARKER"
echo "[install_sdk] 完成，标记: $INSTALLED_MARKER"
