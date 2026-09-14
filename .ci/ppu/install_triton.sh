#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：安装 inductor 用例依赖的 triton，钉版本、且只从内部源装。
#
# 为什么每条测试门禁都要这一步：
#   triton 是 inductor 的代码生成后端，smoke / accuracy / perf / distributed /
#   full_ut 里的 inductor 用例全靠它才跑得起来。PPU 基础镜像虽自带 triton，但那个
#   版本不保证与本 PR 编出来的 torch 匹配 —— torch 侧是按固定的 triton API 写的，
#   版本错位表现为 codegen 阶段 AttributeError / TypeError，而不是干净的 skip。
#
# 必须放在 install_wheel.sh 之后：
#   whl 不把 triton 作为依赖项（见 install_wheel.sh 第 3 段的说明），所以装 torch
#   不会反过来动这里装上的版本；反过来先装 triton 再装 torch 也不会被覆盖，但统一
#   放在装 torch 之后，日志顺序与「先有 torch 再配它的后端」一致，便于排查。
#
# triton 只允许从 TRITON_INDEX（内部 pypi_index 仓）装，这是硬约束：
#   这一版 triton 是内部适配 PPU 的构建，公网 pypi 上的同版本号不是同一份东西。
#   一旦从别处装进来，故障会以 codegen 阶段的怪错误出现，极难定位。因此本脚本
#   刻意与 install_wheel.sh / install_test_deps.sh 不同 —— 不复用 pip_sources.sh
#   的多源回退，也不退回镜像自带的 pip 默认源：
#     - 源为空 -> 直接失败（宁可红，也不要装上一个来源不明的 triton）；
#     - 用 --index-url 指定主源，并把镜像/pod 里既有的 pip 配置隔离掉：
#       PIP_CONFIG_FILE=/dev/null 屏蔽 pip.conf 里可能配着的 extra-index-url，
#       env -u 清掉同名环境变量。这两步都不能省：--index-url 只换主源，并不排除
#       extra 源，留着 extra 源 pip 仍会去那边挑「更合适的」包。
#   代价（已知且有意接受）：内部源抖动、或该版本被下架时，本步骤没有任何回退，
#   门禁会直接失败。
#
# 依赖环境变量：
#   TRITON_INDEX     - 装 triton 的唯一 pip 源（未设则取 PIP_INDEX；两者都为空则失败）
#   PIP_INDEX        - 内部 pip 源，作为 TRITON_INDEX 的默认值
#   TRITON_VERSION   - 钉住的版本（默认 3.6.0）
# =============================================================================
set -euo pipefail

TRITON_VERSION="${TRITON_VERSION:-3.6.0}"
TRITON_INDEX="${TRITON_INDEX:-${PIP_INDEX:-}}"

if [[ -z "${TRITON_INDEX}" ]]; then
    cat >&2 <<'EOF'
[triton][error] TRITON_INDEX / PIP_INDEX 均未设置，无法确定 triton 的来源。
triton 只允许从内部 pypi_index 仓安装（公网 pypi 上的同版本号不是适配 PPU 的那一份），
这里刻意不做默认源兜底。请在 workflow 的 command 里注入 TRITON_INDEX 或 PIP_INDEX。
EOF
    exit 1
fi

echo "=== 安装 triton==${TRITON_VERSION}（唯一源: ${TRITON_INDEX}） ==="
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
    python -m pip install \
    --disable-pip-version-check --no-cache-dir --retries 3 --timeout 60 \
    --index-url "${TRITON_INDEX}" \
    "triton==${TRITON_VERSION}"

# 校验必须在源码树外执行：`python -c` 会把 cwd 放进 sys.path[0]，调用方此时通常正处在
# 源码根目录，未编译的 torch/ 会遮蔽已安装的包（与 install_wheel.sh 同一个坑）。
(cd /tmp && python -c "import triton; print('triton', triton.__version__, triton.__file__)")

echo "[triton] 完成"
