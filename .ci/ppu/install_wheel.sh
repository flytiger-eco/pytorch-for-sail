#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHEEL_DIR="${WHEEL_DIR:-$REPO_ROOT/.ci/ppu/wheelhouse}"
shopt -s nullglob
WHLS=("$WHEEL_DIR"/torch-*.whl)
if [[ ${#WHLS[@]} -eq 0 ]]; then
    cat >&2 <<EOF
[wheel][error] 在 $WHEEL_DIR 下找不到 torch-*.whl。
可能原因：
  - workflow 侧「下载 ppu-ci-810/890 的 build job 产出的 whl」这一步没跑（或下到了别的目录）；
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

if python -m pip uninstall -y torch >/dev/null 2>&1; then
    echo "[wheel] 已卸载环境中原有的 torch"
else
    echo "[wheel] 环境中原本没有 torch（PPU 基础镜像的预期状态）"
fi

source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
ppu_build_pip_candidates

installed=0
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

if torch.__file__.startswith(repo_root):
    sys.exit(f"[wheel] 导入到了源码树里的 torch（{torch.__file__}），whl 未生效")
PY
)

echo "[wheel] 完成"
