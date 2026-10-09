#!/usr/bin/env bash
set -euo pipefail

SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
ENVSETUP="$SDK_INSTALL_DIR/pccl/envsetup.sh"
NCCL_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_NCCL="$NCCL_ENV_SCRIPT_DIR/install_nccl.sh"

if [[ -f "$ENVSETUP" ]]; then
    echo "[nccl_env] 镜像已自带 NCCL，跳过安装: $SDK_INSTALL_DIR/pccl"
else
    echo "[nccl_env] 未找到 NCCL: $ENVSETUP"
    if [[ ! -f "$INSTALL_NCCL" ]]; then
        echo "[nccl_env] 未找到安装脚本: $INSTALL_NCCL" >&2
        exit 1
    fi
    if [[ -z "${NCCL_URL:-}" ]]; then
        echo "[nccl_env] NCCL_URL 未设置，使用 install_nccl.sh 内的兜底默认地址"
    fi
    echo "[nccl_env] 自动执行安装: $INSTALL_NCCL"
    SDK_INSTALL_DIR="$SDK_INSTALL_DIR" NCCL_URL="${NCCL_URL:-}" bash "$INSTALL_NCCL"
    if [[ ! -f "$ENVSETUP" ]]; then
        echo "[nccl_env] 安装后仍未找到: $ENVSETUP" >&2
        exit 1
    fi
fi

set +u
source "$ENVSETUP"
set -u

if [[ -z "${NCCL_HOME:-}" ]]; then
    echo "[nccl_env] NCCL 环境异常: NCCL_HOME 未设置" >&2
    exit 1
fi

export NCCL_HOME
export NCCL_INCLUDE_DIR="${NCCL_HOME}/include"
export NCCL_LIB_DIR="${NCCL_HOME}/lib"

echo "[nccl_env] NCCL 环境就绪: NCCL_HOME=$NCCL_HOME"
