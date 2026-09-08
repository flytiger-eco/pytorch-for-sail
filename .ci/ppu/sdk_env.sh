#!/usr/bin/env bash
# =============================================================================
# 容器内公共 SDK 前置校验：所有 pip install / 编译 / 测试脚本开头先 source 本文件。
# 依赖环境变量：
#   SDK_INSTALL_DIR - SDK 解压目录（默认 /usr/local）
# 行为：
#   1. 检查 $SDK_INSTALL_DIR/PPU_SDK/.ci_installed 标记
#   2. source envsetup.sh
#   3. 校验 nvcc 可用
# =============================================================================
set -euo pipefail

SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
INSTALLED_MARKER="$SDK_INSTALL_DIR/PPU_SDK/.ci_installed"
ENVSETUP="$SDK_INSTALL_DIR/PPU_SDK/envsetup.sh"

if [[ ! -f "$INSTALLED_MARKER" ]]; then
    echo "[sdk_env] SDK 未安装或未落标记: $INSTALLED_MARKER" >&2
    echo "[sdk_env] 请先运行 install_sdk.sh" >&2
    exit 1
fi

if [[ ! -f "$ENVSETUP" ]]; then
    echo "[sdk_env] SDK envsetup.sh 不存在: $ENVSETUP" >&2
    exit 1
fi

set +u
source "$ENVSETUP"
set -u

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[sdk_env] SDK 环境异常: nvcc 不可用" >&2
    exit 1
fi

echo "[sdk_env] SDK 环境就绪: $(nvcc --version | tail -1)"
