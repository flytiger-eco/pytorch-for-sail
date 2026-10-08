#!/usr/bin/env bash
set -euo pipefail

NCCL_URL="${NCCL_URL:-https://pkg.flytiger-eco.com/artifactory/generic-local/NCCL/v2.2.0/nccl-2.27.3-v13_ubuntu2404_sdk-v2.2.0.tar.gz}"
NCCL_INSTALL_DIR="${NCCL_INSTALL_DIR:-/usr/local}"
ENVSETUP="$NCCL_INSTALL_DIR/pccl/envsetup.sh"

if [[ -f "$ENVSETUP" ]]; then
    echo "[install_nccl] 已安装，跳过下载: $NCCL_INSTALL_DIR/pccl"
else
    TARBALL="/tmp/$(basename "$NCCL_URL")"
    echo "[install_nccl] 下载: $NCCL_URL"
    if command -v wget >/dev/null 2>&1; then
        wget -nv -O "$TARBALL" "$NCCL_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 -o "$TARBALL" "$NCCL_URL"
    else
        echo "[install_nccl] 镜像内既无 wget 也无 curl，无法下载 NCCL" >&2; exit 1
    fi

    tar -xf "$TARBALL" -C "$NCCL_INSTALL_DIR"
    rm -f "$TARBALL"
    [[ -f "$ENVSETUP" ]] || {
        echo "[install_nccl] 解压后未找到 $ENVSETUP" >&2; exit 1; }
fi

set +u
source "$ENVSETUP"
set -u
echo "[install_nccl] 完成: NCCL_HOME=${NCCL_HOME:-未设置}"
