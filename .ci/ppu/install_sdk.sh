#!/usr/bin/env bash
set -euo pipefail

DEFAULT_SDK_URL="https://pkg.flytiger-eco.com/artifactory/generic-local/CUDA_SDK/v2.2.0/PPU_SDK_cuda-13.0.0-ubuntu2404-2.2.0-663c22.tar.gz"
SDK_URL="${SDK_URL:-$DEFAULT_SDK_URL}"
SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
ENVSETUP="$SDK_INSTALL_DIR/PPU_SDK/envsetup.sh"
if [[ -f "$ENVSETUP" ]]; then
    echo "[install_sdk] 检测到已安装 SDK，跳过下载: $SDK_INSTALL_DIR/PPU_SDK"
else
    TARBALL="/tmp/$(basename "$SDK_URL")"
    echo "[install_sdk] 下载: $SDK_URL"
    echo "[install_sdk] 落盘: $TARBALL"
    if command -v wget >/dev/null 2>&1; then
        DOWNLOADER=wget
        echo "[install_sdk] 下载器: $(wget --version 2>&1 | head -1)"
    elif command -v curl >/dev/null 2>&1; then
        DOWNLOADER=curl
        echo "[install_sdk] 下载器: $(curl --version 2>&1 | head -1)"
    else
        echo "[install_sdk] 镜像内既无 wget 也无 curl，无法下载 SDK" >&2
        exit 1
    fi
    echo "[install_sdk] 磁盘可用量（SDK 包为 GB 级，/tmp 若是小 tmpfs 会写满）:"
    df -h /tmp "$SDK_INSTALL_DIR" 2>&1 || true

    if [[ "$DOWNLOADER" == wget ]]; then
        wget -nv -O "$TARBALL" "$SDK_URL" 2>&1
    else
        curl -fSL --retry 3 -o "$TARBALL" "$SDK_URL" 2>&1
    fi
    echo "[install_sdk] 下载完成: $(ls -lh "$TARBALL" | awk '{print $5}')"

    echo "[install_sdk] 解压到: $SDK_INSTALL_DIR"
    tar -zxf "$TARBALL" -C "$SDK_INSTALL_DIR"
    rm -f "$TARBALL"
    [[ -f "$ENVSETUP" ]] || {
        echo "[install_sdk] 解压后未找到 $ENVSETUP" >&2; exit 1; }
fi

if [[ ! -e /usr/local/cuda ]]; then
    ln -s "$SDK_INSTALL_DIR/PPU_SDK/CUDA_SDK" /usr/local/cuda
    echo "[install_sdk] 已创建软链: /usr/local/cuda -> PPU_SDK/CUDA_SDK"
else
    echo "[install_sdk] /usr/local/cuda 已存在，跳过软链"
fi

set +u
source "$ENVSETUP"
set -u
echo "[install_sdk] envsetup.sh 已 source"
command -v nvcc && nvcc --version | tail -2 || echo "[install_sdk] 警告: nvcc 不在 PATH" >&2

echo "[install_sdk] 完成: $SDK_INSTALL_DIR/PPU_SDK"
