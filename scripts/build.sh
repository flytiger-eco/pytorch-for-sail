#!/usr/bin/env bash
# build.sh - build and smoke-test the CUDA-compat wheel in a build container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Source tree assembled by setup.sh.
PYTORCH_DIR="${PYTORCH_DIR:-$REPO_ROOT/work/pytorch}"
export PYTORCH_DIR
# Path of the source tree as seen inside the container.  Defaults to the
# host path; override when the container mounts the same filesystem under a
# different prefix (e.g. /mnt/ssd where the host uses /swhome).
CONTAINER_PYTORCH_DIR="${CONTAINER_PYTORCH_DIR:-$PYTORCH_DIR}"
if [ ! -f "$PYTORCH_DIR/pyproject.toml" ] && [ ! -f "$PYTORCH_DIR/setup.py" ]; then
    echo "ERROR: PyTorch source not found at $PYTORCH_DIR. Run scripts/setup.sh first." >&2
    exit 1
fi

# Container configuration.  Pass CONTAINER_NAME to reuse an already-running
# container, or DOCKER_IMAGE to start a fresh one; exactly one is required.
CONTAINER_NAME="${CONTAINER_NAME:-}"
DOCKER_IMAGE="${DOCKER_IMAGE:-}"

# Build options.
: "${USE_SAIL:=0}"
: "${USE_PPU:=1}"
: "${USE_CUDA:=1}"
: "${USE_PCCL:=0}"
: "${USE_NCCL:=1}"
: "${USE_FLASH_ATTENTION:=1}"
: "${BUILD_TEST:=0}"
: "${MAX_JOBS:=128}"
: "${TORCH_CUDA_ARCH_LIST:=8.0a;8.9}"
: "${PYTORCH_SAIL_ARCH:=ppu_10;ppu_15}"
# Heavyweight optional backends, OFF by default to keep validation builds
# fast (FA3 flex flash attention + elementwise optimization).  Export either
# one as 1 to enable the corresponding backend.
: "${USE_FLEX_FLASH_ATTENTION:=0}"
: "${USE_ELEMENTWISE_OPT:=0}"
# Version stamped into the wheel.  The source tree is pinned to the
# v2.14.1 base commit by setup.sh, so the public version is pinned here
# as well; without it the build derives "2.14.1a0+git<sha>" from git.
: "${PYTORCH_BUILD_VERSION:=2.14.1}"
: "${PYTORCH_BUILD_NUMBER:=0}"
# Optional: install the decoupled PCCL/NCCL package from a tarball URL (with
# its sha256) inside the container.  Empty = use the copy pre-installed in
# the base image.
: "${NCCL_PACKAGE_URL:=}"
: "${NCCL_PACKAGE_SHA256:=93513c25af1a5c94ccc38483261b23de6e4912763c3a589e60b1f8dd86cc7a02}"

BUILD_LOG="$REPO_ROOT/build.log"
BUILD_SCRIPT="/tmp/compat_build_in_container.sh"

INNER=$(cat <<'IBSEOF'
#!/usr/bin/env bash
set -euo pipefail
: "${USE_SAIL:=0}"
: "${USE_PPU:=1}"
: "${USE_CUDA:=1}"
: "${USE_PCCL:=0}"
: "${USE_NCCL:=1}"
: "${USE_FLASH_ATTENTION:=1}"
: "${BUILD_TEST:=0}"
: "${MAX_JOBS:=128}"
: "${TORCH_CUDA_ARCH_LIST:=8.0a;8.9}"
: "${PYTORCH_SAIL_ARCH:=ppu_10;ppu_15}"
# Version stamped into the wheel; without it the build derives
# "2.14.1a0+git<sha>" from the git describe of the working tree.
: "${PYTORCH_BUILD_VERSION:=2.14.1}"
: "${PYTORCH_BUILD_NUMBER:=0}"
export USE_SAIL USE_PPU USE_CUDA USE_PCCL USE_NCCL USE_FLASH_ATTENTION BUILD_TEST MAX_JOBS TORCH_CUDA_ARCH_LIST PYTORCH_SAIL_ARCH USE_FLEX_FLASH_ATTENTION USE_ELEMENTWISE_OPT PYTORCH_BUILD_VERSION PYTORCH_BUILD_NUMBER
: "${USE_SYSTEM_NCCL:=1}"
: "${NCCL_INCLUDE_DIR:=/usr/local/pccl/nccl_wrapper/include}"
: "${NCCL_LIB_DIR:=/usr/local/pccl/nccl_wrapper/lib}"
export USE_SYSTEM_NCCL NCCL_INCLUDE_DIR NCCL_LIB_DIR
SRC="$PYTORCH_DIR"
# Avoid picking up the pre-installed broken torch during configure/conversion.
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${NCCL_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$SRC"
echo "=== Environment ==="
python3 --version
python3 -m pip --version
nvcc --version
echo "=== Install build tools ==="
# The SDK base image ships no cmake/ninja; install the toolchain when it is
# missing (idempotent for containers that already provide it).
if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      ccache \
      cmake \
      curl \
      git \
      libffi-dev \
      libgflags-dev \
      libgoogle-glog-dev \
      libjpeg-dev \
      libopenblas-dev \
      libpng-dev \
      libprotobuf-dev \
      libssl-dev \
      libsndfile1-dev \
      ninja-build \
      patch \
      pkg-config \
      protobuf-compiler \
      python3-dev \
      python3-packaging \
      python3-pip \
      python3-pytest \
      python3-requests \
      python3-six \
      python3-typing-extensions \
      python3-yaml
    rm -rf /var/lib/apt/lists/*
fi
cmake --version | head -1
ninja --version
echo "=== Install NCCL/PCCL package ==="
if [ -n "${NCCL_PACKAGE_URL:-}" ]; then
    tmp=$(mktemp -d)
    curl -fL --retry 3 --retry-delay 2 --max-time 900 "$NCCL_PACKAGE_URL" -o "$tmp/pccl.tar.gz"
    echo "$NCCL_PACKAGE_SHA256  $tmp/pccl.tar.gz" | sha256sum -c -
    tar xzf "$tmp/pccl.tar.gz" -C /usr/local
    rm -rf "$tmp"
    echo "PCCL/NCCL package installed from $NCCL_PACKAGE_URL"
else
    echo "Using the PCCL/NCCL copy pre-installed in the base image"
fi
echo "=== Install build deps ==="
# The two packages whose floors exceed the Ubuntu apt versions are overlaid
# with --ignore-installed (apt copies lack RECORD files, so an uninstall
# attempt would fail); they land in site-packages and take precedence.
python3 -m pip install 'scikit-build-core>=1.0' sympy networkx pyyaml six pytest pytest-timeout
python3 -m pip install --ignore-installed 'packaging>=24.2' 'typing-extensions>=4.15,<5'
# The CUDA-compat third-party conversion is managed by the standalone
# tools/cudafy_compat module: it converts flash-attention-for-sail /
# actlize / flex-flash-attention into <build>/cudafy-compat, applies the
# compat-owned patch set (tools/cudafy_compat/patches) and fixes the C++
# standard flags.  It is invoked automatically at CMake configure time;
# re-run it manually with:
#   python3 -m tools.cudafy_compat --output-dir "$SRC/build/cudafy-compat"

echo "=== Build wheel ==="
rm -rf dist
python3 -m pip wheel . --no-build-isolation -w dist
echo "=== Install wheel ==="
# NumPy 2.x ABI is incompatible with the NumPy 1.x used at build time.
python3 -m pip install "numpy<2" --force-reinstall -q
python3 -m pip install --force-reinstall --no-deps dist/torch-2.14.1*.whl
echo "=== Smoke test (matmul) ==="
# Run from /tmp so Python imports the installed wheel, not the source tree.
cd /tmp
# PYTHONPATH served the configure/conversion phase above; drop it so the
# smoke tests import the installed wheel instead of the source tree.
unset PYTHONPATH
python3 - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch file:", torch.__file__)
print("CUDA devices:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available"
assert torch.cuda._is_compiled(), "torch was built without CUDA support"
print("nccl version:", torch.cuda.nccl.version())
a = torch.randn(64, 64, device='cuda')
b = torch.randn(64, 64, device='cuda')
c = a @ b
print("matmul result shape:", c.shape)
assert c.shape == (64, 64)
print("matmul smoke OK")
PY
echo "=== SDPA / transformer CUDA subset ==="
# Bounded subset: the full `-k cuda` selection runs ~12k tests single-
# process and takes hours; basic + sdpa covers the core paths quickly.
python3 -m pytest "$SRC"/test/test_transformers.py -k "test_basic or test_sdpa" -x -q --timeout=300 || true
echo "=== NCCL tests ==="
# Upstream test file is missing an unittest import in some revisions.
if ! grep -q "^import unittest" "$SRC"/test/distributed/test_nccl.py; then
    sed -i '5a import unittest' "$SRC"/test/distributed/test_nccl.py
fi
# Exclude the whole NCCLSymmetricMemory* family: on PPU the one-shot
# collectives need multimem/NVLS hardware (torch 2.10 verified:
# test_symmetric_memory.py = 35 passed / 66 skipped, all skips are
# multimem/multicast) and the NCCL-backend rendezvous fails fast on
# NCCL 2.27 (needs >= 2.28), so they add nothing to the smoke run.
# Substring -k is used on purpose: "not NCCLSymmetricMemoryTest" would
# miss NCCLSymmetricMemoryNccl2Test / NCCLSymmetricMemoryNcclLazyTest.
# CUDA_VISIBLE_DEVICES=0,1: with all 16 devices visible, spawn-based
# tests start 16 workers whose PCCL init hangs the PPU driver (D-state,
# unkillable, semaphore leak, blocks later CUDA context init).  Two
# devices keep the spawn small; classic collectives still exercise the
# real multi-GPU path.  -x/--timeout/outer timeout stay as hard bounds.
CUDA_VISIBLE_DEVICES=0,1 timeout 900 python3 -m pytest "$SRC"/test/distributed/test_nccl.py \
    -k "not NCCLSymmetricMemory" -x -q --timeout=300 || true
echo "=== Build and test finished ==="
IBSEOF
)
cat > "$BUILD_SCRIPT" <<< "$INNER"
chmod +x "$BUILD_SCRIPT"

rm -f "$BUILD_LOG"

if [ -z "$CONTAINER_NAME" ] && [ -z "$DOCKER_IMAGE" ]; then
    echo "ERROR: set CONTAINER_NAME (reuse a running container) or DOCKER_IMAGE (start a new one)." >&2
    exit 1
fi

if [ -n "$CONTAINER_NAME" ]; then
    echo "Running build in existing container $CONTAINER_NAME"
    docker cp "$BUILD_SCRIPT" "$CONTAINER_NAME:/build.sh"
    docker exec \
        -e USE_SAIL -e USE_PPU -e USE_CUDA -e USE_PCCL -e USE_NCCL \
        -e USE_FLASH_ATTENTION \
        -e USE_SYSTEM_NCCL -e NCCL_INCLUDE_DIR -e NCCL_LIB_DIR \
        -e BUILD_TEST -e MAX_JOBS -e TORCH_CUDA_ARCH_LIST -e PYTORCH_SAIL_ARCH \
        -e USE_FLEX_FLASH_ATTENTION -e USE_ELEMENTWISE_OPT \
        -e PYTORCH_BUILD_VERSION -e PYTORCH_BUILD_NUMBER \
        -e NCCL_PACKAGE_URL -e NCCL_PACKAGE_SHA256 \
        -e PYTORCH_DIR="$CONTAINER_PYTORCH_DIR" \
        "$CONTAINER_NAME" bash /build.sh | tee -a "$BUILD_LOG"
else
    echo "Running build in new container from $DOCKER_IMAGE"
    # PPU hosts expose the accelerator via /dev/alixpu* (control node +
    # per-card nodes), not via the NVIDIA runtime; pass the control and
    # card devices through so smoke tests can touch the hardware.  Extend
    # with more --device entries when multi-card tests are needed.
    docker run --rm \
        --device /dev/alixpu \
        --device /dev/alixpu_ctl \
        -v "$PYTORCH_DIR:$PYTORCH_DIR" \
        -v "$BUILD_SCRIPT:/build.sh" \
        -e USE_SAIL -e USE_PPU -e USE_CUDA -e USE_PCCL -e USE_NCCL \
        -e USE_FLASH_ATTENTION \
        -e USE_SYSTEM_NCCL -e NCCL_INCLUDE_DIR -e NCCL_LIB_DIR \
        -e BUILD_TEST -e MAX_JOBS -e TORCH_CUDA_ARCH_LIST -e PYTORCH_SAIL_ARCH \
        -e USE_FLEX_FLASH_ATTENTION -e USE_ELEMENTWISE_OPT \
        -e PYTORCH_BUILD_VERSION -e PYTORCH_BUILD_NUMBER \
        -e NCCL_PACKAGE_URL -e NCCL_PACKAGE_SHA256 \
        -e PYTORCH_DIR \
        "$DOCKER_IMAGE" bash /build.sh | tee -a "$BUILD_LOG"
fi

echo ""
echo "Build log: $BUILD_LOG"
