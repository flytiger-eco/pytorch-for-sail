#!/usr/bin/env bash
# =============================================================================
# 容器内：ccache 启用与配置。
#   - 独立执行：把 ccache 装好、把配置写进 $CCACHE_DIR/ccache.conf（供 workflow 预置步骤用）
#   - 被 source（build_wheel.sh）：额外导出 CCACHE_DIR 与 CMAKE_*_COMPILER_LAUNCHER
#
# 为什么还要显式导出 CMAKE_*_COMPILER_LAUNCHER：
#   CMakeLists.txt 的 USE_CCACHE 分支会自己 find_program(ccache) 并设置三个 launcher，
#   但该 option 可被上层关掉；显式导出同名环境变量（CMake 会用它初始化对应 cache 变量）
#   可以避免"以为在用缓存、实际在裸编译"这种最难发现的退化。
#
# 依赖环境变量：
#   CCACHE_DIR           - 缓存目录（默认 /root/.cache/ccache，由宿主机挂载进来）
#   CCACHE_MAXSIZE       - 缓存上限（默认 5G；GitHub 单仓库缓存总配额 10GiB）
#   CCACHE_SLOPPINESS    - 命中宽松度，默认见下方说明
#   CCACHE_COMPILERCHECK - 编译器身份判定方式（默认 content）
#   PPU_CCACHE_REQUIRED  - 1 = ccache 不可用就直接失败；默认 0（只告警，退化成无缓存编译）
# =============================================================================
set -euo pipefail

CCACHE_DIR="${CCACHE_DIR:-/root/.cache/ccache}"
CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-5G}"
# time_macros            : third_party 里有用到 __DATE__/__TIME__ 的 TU，默认策略拒绝缓存它们
# include_file_mtime/ctime: CI 每轮都是全新签出，codegen 头文件更是编译过程中现场生成，
#                          mtime 永远是"刚刚"。ccache 默认会把"太新"的头文件判为不可缓存，
#                          不放宽这两项，含生成头文件的目标（几乎是全部）命中率直接归零
CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-time_macros,include_file_mtime,include_file_ctime}"
# 镜像固定，但 PPU SDK 是运行时安装的，nvcc 每轮的 mtime 都不一样；
# 默认 compiler_check=mtime 会让所有 CUDA 目标全部 miss，必须按内容哈希
CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-content}"
PPU_CCACHE_REQUIRED="${PPU_CCACHE_REQUIRED:-0}"

log() {
    echo "[ccache] $*"
}

# ---- 1. 确保 ccache 存在（镜像自带优先，缺失才联网装） ----
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
    # 不阻断：没有缓存只是慢，产物仍然正确
    log "警告: ccache 不可用，本轮退化成无缓存编译" >&2
    export PPU_CCACHE_ENABLED=0
else
    # ---- 2. 写配置 ----
    # 直接写配置文件而不用 `ccache --set-config`：语法在 3.x / 4.x 间有差异，
    # 而 $CCACHE_DIR/ccache.conf 是所有版本都读的路径。配置随缓存目录一起进 GitHub
    # cache，恢复后无需重新设置也能保持一致。
    mkdir -p "$CCACHE_DIR"
    cat > "$CCACHE_DIR/ccache.conf" <<EOF
max_size = ${CCACHE_MAXSIZE}
sloppiness = ${CCACHE_SLOPPINESS}
compiler_check = ${CCACHE_COMPILERCHECK}
EOF

    export CCACHE_DIR
    # ccache 的命中判定只看预处理后的内容与编译命令，与源码 mtime 无关，
    # 所以跨 run / 跨 PR 复用不需要任何 mtime 对齐处理
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    export PPU_CCACHE_ENABLED=1

    log "$(ccache --version | head -1)"
    log "dir=${CCACHE_DIR} max_size=${CCACHE_MAXSIZE} compiler_check=${CCACHE_COMPILERCHECK}"
    log "sloppiness=${CCACHE_SLOPPINESS}"
    ccache -s 2>/dev/null | sed 's/^/[ccache]   /' || true
fi
