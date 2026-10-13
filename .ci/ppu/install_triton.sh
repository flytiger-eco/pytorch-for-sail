#!/usr/bin/env bash
set -euo pipefail

TRITON_VERSION="${TRITON_VERSION:-3.7.0}"
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

(cd /tmp && python -c "import triton; print('triton', triton.__version__, triton.__file__)")

echo "[triton] 完成"
