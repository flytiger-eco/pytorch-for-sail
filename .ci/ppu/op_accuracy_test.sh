#!/usr/bin/env bash
# PPU operator accuracy comparison suite (pputests/).
#
# Runs the config-driven golden-vs-PPU accuracy framework. Default is the smoke
# gate (fast, CI-friendly); pass LEVEL=full for the nightly full matrix.
#
# Env:
#   LEVEL       smoke|full   (default smoke)
#   DTYPES      comma list   (default bf16,fp32; add fp16 to opt in)
#   JOBS        parallel threads for full mode (default 4)
#   OPS         optional exact operator names (e.g. relu,softmax; sum for Sum)
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .ci/ppu/sdk_env.sh
bash .ci/ppu/install_wheel.sh

LEVEL="${LEVEL:-smoke}"
DTYPES="${DTYPES:-bf16,fp32}"
JOBS="${JOBS:-4}"

echo "=== 环境自检 (PPU) ==="
python --version
(cd /tmp && python -c "import sys, torch; available = torch.cuda.is_available(); count = torch.cuda.device_count(); print('torch', torch.__version__, torch.__file__); print('cuda_available', available); print('device_count', count); sys.exit(0 if available and count > 0 else 1)")
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"

echo "=== PPU 算子精度对比 (level=${LEVEL} dtypes=${DTYPES}) ==="
cd pputests
if [ -n "${OPS:-}" ]; then
    python run.py --ops "$OPS" --dtypes "$DTYPES" --level "$LEVEL" -n "$JOBS" \
        --report markdown,csv,json,junit --out reports/
else
    python run.py --all --dtypes "$DTYPES" --level "$LEVEL" -n "$JOBS" \
        --report markdown,csv,json,junit --out reports/
fi

echo "[op-accuracy] 完成，报告见 pputests/reports/"
