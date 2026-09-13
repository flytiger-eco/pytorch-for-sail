#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：安装 ppu-linux-build 编译产出的 torch whl。
#
# 为什么必须有这一步：
#   smoke / accuracy 两条门禁改用 PPU 基础镜像（pkg.../ppu:...）后，镜像里只有
#   PPU SDK 与 triton，没有 torch。被测的 torch 必须是本 PR 的编译产物，否则门禁
#   测的是镜像里那份与 PR 无关的旧 torch，等于没测。
#   whl 由 workflow 侧从 ppu-linux-build 的 artifact 下载到 WHEEL_DIR（默认在源码树
#   内），再随源码一起被 ppu-distributed-action 打包送进 pod。
#
# 与 install_test_deps.sh 的分工：本脚本只管 torch 本体及其运行时依赖
# （filelock / typing-extensions / setuptools / sympy / networkx / jinja2 / fsspec，
# 见 setup.py 的 install_requires），测试框架依赖仍由 install_test_deps.sh 负责。
#
# 依赖环境变量：
#   WHEEL_DIR           - whl 所在目录（默认 <repo>/.ci/ppu/wheelhouse）
#   PIP_INDEX           - 首选 pip 源（可选；不设则直接用镜像自带配置）
#   PIP_INDEX_FALLBACKS - 空格分隔的备用源（可选）
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHEEL_DIR="${WHEEL_DIR:-$REPO_ROOT/.ci/ppu/wheelhouse}"

# -----------------------------------------------------------------------------
# 1) 定位 whl
# -----------------------------------------------------------------------------
shopt -s nullglob
WHLS=("$WHEEL_DIR"/torch-*.whl)
if [[ ${#WHLS[@]} -eq 0 ]]; then
    cat >&2 <<EOF
[wheel][error] 在 $WHEEL_DIR 下找不到 torch-*.whl。
可能原因：
  - workflow 侧「下载 ppu-linux-build 产出的 whl」这一步没跑（或下到了别的目录）；
  - ppu-distributed-action 打包源码时漏掉了这个 whl —— 该目录必须在源码树内，
    且不能被 .gitignore 命中（这也是这里用 .ci/ppu/wheelhouse 而不是 dist/ 的原因，
    dist/ 在 .gitignore 里，按 gitignore 过滤的打包方式会把它整个丢掉）。
当前目录内容：
EOF
    ls -la "$WHEEL_DIR" >&2 2>/dev/null || echo "  （目录不存在）" >&2
    exit 1
fi
if [[ ${#WHLS[@]} -gt 1 ]]; then
    echo "[wheel][error] $WHEEL_DIR 下有多个 torch whl，无法确定该装哪个：" >&2
    printf '  %s\n' "${WHLS[@]}" >&2
    exit 1
fi
WHEEL="${WHLS[0]}"
echo "=== 安装 PPU torch whl ==="
echo "[wheel] 包: $(basename "$WHEEL") ($(du -h "$WHEEL" | cut -f1))"

# -----------------------------------------------------------------------------
# 2) 先卸干净再装
# 直接 pip install 一个本地 whl 时，若环境里已有同版本 torch，pip 可能判为
# "Requirement already satisfied" 而跳过 —— 那样跑的就是旧 torch，且日志上几乎看不出来。
# 显式先卸载，把「装的是哪一份」变成确定行为。
# -----------------------------------------------------------------------------
if python -m pip uninstall -y torch >/dev/null 2>&1; then
    echo "[wheel] 已卸载环境中原有的 torch"
else
    echo "[wheel] 环境中原本没有 torch（PPU 基础镜像的预期状态）"
fi

# -----------------------------------------------------------------------------
# 3) 逐个源尝试安装
# 只有 torch 的那几个纯 python 依赖需要联网（triton 不是 whl 的依赖项，由镜像自带，
# 所以这里绝不能用 --force-reinstall：那会连带重装/降级镜像里适配好的 triton 等包）。
# --no-cache-dir：whl 有 GB 级，别在 pod 的磁盘上再存一份。
# -----------------------------------------------------------------------------
# 候选源列表与 install_test_deps.sh 共用一份，见 .ci/ppu/pip_sources.sh。
# 关键：本脚本跑在 install_test_deps.sh 之前，必须自带同样的多源回退 —— 否则首选源
# （pypi_index 这个 VIRTUAL 仓）在 pod 内偶发解析成 "from versions: none" 时，本脚本
# 会直接失败，根本轮不到后面那个带回退的脚本，表现为 smoke 过、accuracy 挂。
# shellcheck source=.ci/ppu/pip_sources.sh
source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
ppu_build_pip_candidates

installed=0
# 末尾追加 __pip_default__：所有内网/公网候选都失败时，退回镜像自带的 pip 配置兜底。
for index in "${PIP_CANDIDATES[@]}" __pip_default__; do
    pip_args=(--disable-pip-version-check --no-cache-dir --retries 1 --timeout 20)
    if [[ "${index}" == "__pip_default__" ]]; then
        label="pip 默认源"
    else
        label="${index}"
        pip_args+=(-i "${index}")
    fi
    echo "[wheel] 尝试源: ${label}"
    if python -m pip install "${pip_args[@]}" "$WHEEL"; then
        echo "[wheel] 安装成功（源: ${label}）"
        installed=1
        break
    fi
    echo "[wheel][warn] 源 ${label} 失败，换下一个"
done

if [[ "${installed}" -ne 1 ]]; then
    cat >&2 <<'EOF'
[wheel][error] whl 装不上。torch 本体是本地文件、不走网络，失败几乎只可能出在它的
运行时依赖（filelock / typing-extensions / setuptools / sympy / networkx / jinja2 /
fsspec）取不到，或磁盘写满。pip 源的连通性判断方法见 install_test_deps.sh 的
「pip 源连通性诊断」段落。
EOF
    exit 1
fi

# -----------------------------------------------------------------------------
# 4) 校验：必须导入到 site-packages 里那份
# 校验只能在源码树外执行：`python -c` 会把 cwd 放进 sys.path[0]，在源码根目录下
# 未编译的 torch/ 会遮蔽刚装好的包，报 ModuleNotFoundError: No module named 'torch.version'。
# 顺带打印 git_version，用于确认这份 whl 确实是本 PR 的 commit 编出来的。
# -----------------------------------------------------------------------------
(
    cd /tmp
    python - "$REPO_ROOT" <<'PY'
import sys

import torch

repo_root = sys.argv[1]
print("torch", torch.__version__, torch.__file__)
print("git_version", getattr(torch.version, "git_version", "<unknown>"))
print("built_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())

# 导入到源码树里那份未编译的 torch/ 时，后面所有用例都会以各种奇怪方式失败，
# 在这里就断掉，比让 run_test.py 报一堆无关错误好定位
if torch.__file__.startswith(repo_root):
    sys.exit(f"[wheel] 导入到了源码树里的 torch（{torch.__file__}），whl 未生效")
PY
)

echo "[wheel] 完成"
