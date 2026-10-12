#!/usr/bin/env bash
set -euo pipefail

SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
ENVSETUP="$SDK_INSTALL_DIR/PPU_SDK/envsetup.sh"
SDK_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SDK="$SDK_ENV_SCRIPT_DIR/install_sdk.sh"

if [[ -f "$ENVSETUP" ]]; then
    echo "[sdk_env] 镜像已自带 SDK，跳过安装: $SDK_INSTALL_DIR/PPU_SDK"
else
    echo "[sdk_env] 未找到 SDK: $ENVSETUP"
    if [[ ! -f "$INSTALL_SDK" ]]; then
        echo "[sdk_env] 未找到安装脚本: $INSTALL_SDK" >&2
        exit 1
    fi
    if [[ -z "${SDK_URL:-}" ]]; then
        echo "[sdk_env] SDK_URL 未设置，使用 install_sdk.sh 内的兜底默认地址"
    fi
    echo "[sdk_env] 自动执行安装: $INSTALL_SDK"
    SDK_INSTALL_DIR="$SDK_INSTALL_DIR" SDK_URL="${SDK_URL:-}" bash "$INSTALL_SDK"
    if [[ ! -f "$ENVSETUP" ]]; then
        echo "[sdk_env] 安装后仍未找到: $ENVSETUP" >&2
        exit 1
    fi
fi

set +u
source "$ENVSETUP"
set -u

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[sdk_env] SDK 环境异常: nvcc 不可用" >&2
    exit 1
fi

echo "[sdk_env] SDK 环境就绪: $(nvcc --version | tail -1)"
