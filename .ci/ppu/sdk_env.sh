#!/usr/bin/env bash
# =============================================================================
# 容器内公共 SDK 前置校验：所有 pip install / 编译 / 测试脚本开头先 source 本文件。
# 依赖环境变量：
#   SDK_INSTALL_DIR - SDK 解压目录（默认 /usr/local）
#   SDK_URL          - SDK 包地址（可选，缺省用 install_sdk.sh 内的兜底默认地址）
# 行为：
#   1. 检查 $SDK_INSTALL_DIR/PPU_SDK/envsetup.sh，不存在则调用 install_sdk.sh 安装
#   2. source envsetup.sh
#   3. 校验 nvcc 可用
# =============================================================================
set -euo pipefail

SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
ENVSETUP="$SDK_INSTALL_DIR/PPU_SDK/envsetup.sh"
SDK_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SDK="$SDK_ENV_SCRIPT_DIR/install_sdk.sh"

# 判据直接用 envsetup.sh 是否存在，而不是 install_sdk.sh 落的 .ci_installed 标记：
# 后者只有本仓库脚本会写，在任何全新镜像上必然不存在，会让自带 SDK 的 PPU
# 发布镜像每次都白跑一次 GB 级下载（也是之前 install_sdk 下载失败的起因）。
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
