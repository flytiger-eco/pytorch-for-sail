#!/usr/bin/env bash
set -euo pipefail

CCACHE_DIR="${CCACHE_DIR:-/root/.cache/ccache}"
CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-5G}"
CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-time_macros,include_file_mtime,include_file_ctime}"
CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-content}"
PPU_CCACHE_REQUIRED="${PPU_CCACHE_REQUIRED:-0}"

log() {
    echo "[ccache] $*"
}

if ! command -v ccache >/dev/null 2>&1; then
    log "镜像未自带 ccache，尝试 apt 安装"
    for attempt in 1 2; do
        if apt-get update >/dev/null 2>&1 \
            && apt-get install -y --no-install-recommends ccache >/dev/null 2>&1; then
            break
        fi
        log "第 ${attempt} 次安装失败" >&2
        sleep 5
    done
fi

if ! command -v ccache >/dev/null 2>&1; then
    if [[ "$PPU_CCACHE_REQUIRED" == "1" ]]; then
        log "错误: PPU_CCACHE_REQUIRED=1 但 ccache 装不上" >&2
        exit 1
    fi
    log "警告: ccache 不可用，本轮退化成无缓存编译" >&2
    export PPU_CCACHE_ENABLED=0
else
    mkdir -p "$CCACHE_DIR"
    cat > "$CCACHE_DIR/ccache.conf" <<EOF
max_size = ${CCACHE_MAXSIZE}
sloppiness = ${CCACHE_SLOPPINESS}
compiler_check = ${CCACHE_COMPILERCHECK}
EOF

    export CCACHE_DIR
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    export PPU_CCACHE_ENABLED=1

    log "$(ccache --version | head -1)"
    log "dir=${CCACHE_DIR} max_size=${CCACHE_MAXSIZE} compiler_check=${CCACHE_COMPILERCHECK}"
    log "sloppiness=${CCACHE_SLOPPINESS}"
    ccache -s 2>/dev/null | sed 's/^/[ccache]   /' || true
fi
