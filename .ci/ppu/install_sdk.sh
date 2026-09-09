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
ENVSETUP="$SDK_INSTALL_DIR/PPU_SDK/envsetup.sh"

# ---- 幂等：envsetup.sh 已存在则只做自检 ----
# 判据用 envsetup.sh 而不是自建的 .ci_installed 标记：标记只有本脚本会写，
# PPU 发布镜像自带 SDK 时它必然不存在，会白跑一次 GB 级下载。
if [[ -f "$ENVSETUP" ]]; then
    echo "[install_sdk] 检测到已安装 SDK，跳过下载: $SDK_INSTALL_DIR/PPU_SDK"
else
    TARBALL="/tmp/$(basename "$SDK_URL")"
    echo "[install_sdk] 下载: $SDK_URL"
    echo "[install_sdk] 落盘: $TARBALL"

    # 下载前先把环境交代清楚：旧写法 wget -q 把错误全压掉，而 CI 只回传 stdout，
    # 一旦失败日志里看不到任何原因。
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

    # 不用 -q/--show-progress：--show-progress 是 GNU wget 1.16+ 的选项，busybox wget
    # 不认会直接退出；-q 又会把 404/401/连不上 的原因一并吞掉。
    # -nv / -sS 既不刷进度条，又保留错误；2>&1 是为了让错误进得了 CI 日志。
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

# ---- 软链 CUDA_SDK -> /usr/local/cuda（已存在则跳过） ----
if [[ ! -e /usr/local/cuda ]]; then
    ln -s "$SDK_INSTALL_DIR/PPU_SDK/CUDA_SDK" /usr/local/cuda
    echo "[install_sdk] 已创建软链: /usr/local/cuda -> PPU_SDK/CUDA_SDK"
else
    echo "[install_sdk] /usr/local/cuda 已存在，跳过软链"
fi

# ---- 自检：envsetup.sh 生效 + nvcc 可用 ----
# 本脚本是被 bash 子进程调用的，这里 source 只影响自己，仅作安装后的验证；
# 真正给测试/编译进程生效的 source 在 sdk_env.sh 里。
set +u
source "$ENVSETUP"
set -u
echo "[install_sdk] envsetup.sh 已 source"
command -v nvcc && nvcc --version | tail -2 || echo "[install_sdk] 警告: nvcc 不在 PATH" >&2

echo "[install_sdk] 完成: $SDK_INSTALL_DIR/PPU_SDK"
