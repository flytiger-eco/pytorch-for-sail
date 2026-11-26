# PyTorch-for-SAIL CUDA-Compatible Build

Build from one dedicated workspace. Clone the public upstream PyTorch sources inside this deployment directory, convert the required third-party code **in place**, and build the wheel from the same source directory.

```text
pytorch-cuda-compat/              # Deployment repository
├── patches/                     # CUDA-compatibility patches
├── scripts/                     # Optional helpers; not required below
└── pytorch/                     # Upstream clone and build working directory
    ├── third_party/             # cudafy converts the required dependencies here
    ├── build/                   # CMake/Ninja build files
    ├── dist/                    # Torch wheel and its SHA256 checksum
    └── compile.log              # Wheel build log
```

Requirements: Linux x86_64, a working Zhenwu PPU driver, Docker access, network access to the image and Git repositories, and sufficient disk space and RAM for a full PyTorch build. The example targets M890 with PPU SDK 2.2.1 / CUDA 13.0 and Python 3.12. Reduce `MAX_JOBS` on smaller machines.

Use a fresh workspace and an unused container name. Do not reuse an existing build tree or overwrite another user's container. Run each section in order, in the same Bash session for that environment. `set -e` exits that shell on a command failure; inspect the error before retrying.

The README and patches must come from the same deployment revision, including `--inplace` support. Inplace conversion modifies FlashAttention, actlize, and FlexFA under `pytorch/third_party/`; it does **not** convert the entire Torch tree. Keep a separate pristine checkout for SAIL builds. Do not run `tools.sailify` or update/reset submodules after conversion.

## 1. Prepare the sources (host)

```bash
set -euo pipefail

# Skip these first two commands if you already have this deployment revision.
git clone --depth 1 --single-branch -b feature/v2.14.1-dev \
  https://github.com/flytiger-eco/pytorch-for-sail.git pytorch-cuda-compat
cd pytorch-cuda-compat
export COMPAT_ROOT="$PWD"

# Clone upstream inside the workspace, not from a local development fork.
git clone --depth 1 --single-branch -b v2.14.1 \
  https://github.com/t-head/pytorch-for-sail.git pytorch
cd pytorch
git fetch --depth 1 --no-tags --no-recurse-submodules \
  https://github.com/t-head/pytorch-for-sail.git refs/heads/v2.14.1
UPSTREAM_HEAD=$(git rev-parse --verify 'FETCH_HEAD^{commit}')
git checkout --detach "$UPSTREAM_HEAD"
git show -s --format='%H %s' HEAD

# Add the cudafy dependency first, at the exact commit recorded by patch 0001.
git apply --check --index "$COMPAT_ROOT"/patches/0001-*.patch
git apply --index "$COMPAT_ROOT"/patches/0001-*.patch
git submodule update --init third_party/cudafy-for-sail
test "$(git -C third_party/cudafy-for-sail rev-parse HEAD)" = \
  9d19a24573b8f0405fcd28770f7dee7d53901aac

# Apply the build/conversion changes and initialize all nested dependencies.
git apply --check --index "$COMPAT_ROOT"/patches/000[234]-*.patch
git apply --index "$COMPAT_ROOT"/patches/000[234]-*.patch
git submodule sync --recursive
git submodule update --init --recursive --jobs 8
git diff --cached --stat
```

Record the printed upstream SHA. The upstream branch is fetched anew, not pinned to a historical build. Stop on fetch or patch-check failure; do not substitute an older revision. Patches are pure diffs: `git apply --index` stages them without creating commits. FlashAttention and actlize retain the upstream gitlinks; do not use `submodule update --remote`.

## 2. Open a dedicated SDK container (host)

The entire workspace is mounted at `/workspace/pytorch-cuda-compat`. All remaining build commands run in `/workspace/pytorch-cuda-compat/pytorch`.

```bash
SDK_IMAGE=pkg.flytiger-eco.com/docker_release/ppu:sdk2.2.1-cuda13.0-ubuntu24-py312-20261103

# Include card nodes as well as control nodes. Restrict this list on shared hosts.
DEVICE_ARGS=()
for device in /dev/alixpu*; do
  if [ -e "$device" ]; then DEVICE_ARGS+=(--device "$device"); fi
done

docker pull "$SDK_IMAGE"
docker run --name torch-cuda-compat-build -dit --entrypoint /bin/bash --shm-size=8g \
  "${DEVICE_ARGS[@]}" \
  -v "$COMPAT_ROOT:/workspace/pytorch-cuda-compat" \
  -w /workspace/pytorch-cuda-compat/pytorch "$SDK_IMAGE"
docker exec -it torch-cuda-compat-build bash
```

If the name already exists, check its purpose and mounts before reusing it, or choose a different name. The background container remains running after you leave its interactive shell.

## 3. Install dependencies and set the environment (container)

Install dependencies once in this dedicated container:

```bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates git ccache cmake ninja-build patch pkg-config \
  python3-dev libffi-dev libssl-dev libopenblas-dev libjpeg-dev libpng-dev \
  libgflags-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler libsndfile1-dev
python3 -m pip install 'numpy<2' 'scikit-build-core>=1.0' \
  'setuptools>=77.0.3,<82' 'sympy>=1.13.3' 'networkx>=2.5.1' \
  filelock jinja2 'fsspec>=0.8.5' pyyaml six requests pytest pytest-timeout

# Overlay apt-managed packages without attempting to uninstall their copies.
python3 -m pip install --ignore-installed \
  'packaging>=24.2' 'typing-extensions>=4.15,<5'
# This image includes MKL 2021.1.1; supply its matching runtime dependencies.
python3 -m pip install 'intel-openmp==2021.1.1' 'tbb==2021.1.1'
python3 -m pip check
```

Run the following environment block in every new container shell. It also restores the build directory, so there is no need to reinstall dependencies when reconnecting.

```bash
set -eo pipefail
cd /workspace/pytorch-cuda-compat/pytorch
export PIP_BREAK_SYSTEM_PACKAGES=1
source /usr/local/PPU_SDK/envsetup.sh
source /usr/local/pccl/envsetup.sh
export CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK
export CUDA_PATH="$CUDA_HOME" CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc" PATH="$CUDA_HOME/bin:$PATH"

# PCCL provides the NCCL-compatible headers and shared library.
export NCCL_INCLUDE_DIR=/usr/local/pccl/nccl_wrapper/include
export NCCL_LIB_DIR=/usr/local/pccl/nccl_wrapper/lib
export LD_LIBRARY_PATH="$NCCL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test -f "$NCCL_INCLUDE_DIR/nccl.h"
test -e "$NCCL_LIB_DIR/libnccl.so"

export USE_PPU=1 USE_CUDA=1 USE_SAIL=0
export USE_PCCL=0 USE_NCCL=1 USE_SYSTEM_NCCL=1
export USE_FLASH_ATTENTION=1 USE_FLEX_FLASH_ATTENTION=0 USE_ELEMENTWISE_OPT=0
export BUILD_TEST=0 MAX_JOBS=32
export TORCH_CUDA_ARCH_LIST="8.9" PYTORCH_SAIL_ARCH="ppu_15"
export PYTORCH_BUILD_VERSION=2.14.1 PYTORCH_BUILD_NUMBER=0

# Use third_party directly, with no separate cudafy-derived source tree.
unset TORCH_COMPAT_DERIVED_DIR
export TORCH_COMPAT_INPLACE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The actual device target is `ppu0015`; CUDA naming does not imply NVIDIA hardware. Stop if the NCCL checks fail and install the package matching the SDK. For other images, match their own MKL dependencies. Changing architectures requires a fresh build directory; this example targets only M890.

## 4. Convert in place and build the wheel (container)

Stay in the `pytorch` directory:

```bash
# Convert third-party sources once, apply compatibility fixes, and select C++20.
python3 -m tools.cudafy_compat --inplace

# Preserve existing wheels instead of silently overwriting them.
mkdir -p dist
WHEELS=(dist/torch-*.whl)
if [ -e "${WHEELS[0]}" ]; then
  echo "dist already contains a Torch wheel; preserve it before rebuilding." >&2
  exit 1
fi

# pipefail preserves a build failure even though output also goes through tee.
python3 -m pip wheel . -v --no-build-isolation --no-deps --no-cache-dir \
  --wheel-dir dist 2>&1 | tee compile.log

WHEELS=(dist/torch-*.whl)
test "${#WHEELS[@]}" -eq 1
test -f "${WHEELS[0]}"
WHEEL=$(realpath "${WHEELS[0]}")
sha256sum "$WHEEL" | tee "$WHEEL.sha256"
python3 - "$WHEEL" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    assert archive.testzip() is None, "wheel CRC validation failed"
print("wheel CRC: OK")
PY
```

CMake uses the same inplace entry point and skips conversion when its recorded inputs and converted sources are unchanged. The receipt is `pytorch/.cudafy-compat-inplace.json`. If conversion fails partway, or submodules, cudafy, or conversion patches change afterward, preserve your changes and assemble a fresh checkout. Do not delete the receipt to retry or use `--force` in inplace mode. A compilation failure alone can be retried after fixing the build environment, provided the converted inputs remain unchanged.

Only continue after `Successfully built torch`, exit status 0, and `wheel CRC: OK`.

The wheel is **`pytorch-cuda-compat/pytorch/dist/torch-2.14.1-cp312-cp312-linux_x86_64.whl`** on the host, or **`/workspace/pytorch-cuda-compat/pytorch/dist/`** inside the container. The `.whl.sha256` file is beside it; `compile.log` and `build/` stay in the same source directory. The deployment repository ignores the nested `pytorch/` tree; sources, logs, and wheels are not automatically included in deployment commits.

## 5. Install and verify (container)

```bash
# Select the wheel again so this block also works after reconnecting.
WHEELS=(dist/torch-*.whl)
test "${#WHEELS[@]}" -eq 1
test -f "${WHEELS[0]}"
WHEEL=$(realpath "${WHEELS[0]}")
unset PYTHONPATH
python3 -m pip install --force-reinstall --no-deps "$WHEEL"
python3 -m pip check

# -I avoids importing the source checkout while staying in the build directory.
python3 -I - <<'PY'
from pathlib import Path
import torch
print(torch.__version__, torch.__file__, torch._C.__file__)
assert torch.__version__ == "2.14.1"
assert torch.version.cuda == "13.0" and torch.cuda._is_compiled()
assert Path(torch.__file__).parent == Path(torch._C.__file__).parent
print("installed wheel import: OK")
PY
```

Inspect the cards separately before testing:

```bash
ppu-smi
```

Choose a card with no other user processes, no workload memory, and zero utilization. Zero utilization alone is not sufficient. Do not terminate other users' jobs. Replace physical card `0` below with your idle card; it becomes logical device `0` inside Python.

```bash
CUDA_VISIBLE_DEVICES=0 timeout -k 10s 180s python3 -I - <<'PY'
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

assert torch.cuda.is_available()
a = torch.randn(64, 64, device="cuda")
assert torch.isfinite(a @ a.T).all()
# SDPA layout: [batch, heads, sequence length, head dimension].
q, k, v = [torch.randn(1, 8, 128, 64, device="cuda",
                      dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
out.sum().backward()
torch.cuda.synchronize()
assert torch.isfinite(out).all()
assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))
print("matmul / Flash forward / backward: OK")
PY
```

This is a finite-value and synchronization smoke test, not full accuracy, performance, or unit-test acceptance. The wheel is non-editable but still requires the matching PPU SDK, NCCL wrapper, and system/MKL libraries; it is not a self-contained NVIDIA CUDA wheel.

To reconnect, use `docker exec -it torch-cuda-compat-build bash` (`docker start torch-cuda-compat-build` first if stopped), then rerun only the environment block in step 3. Step 5 selects the wheel from `dist/` again and does not depend on `$WHEEL` surviving from an earlier shell.

## Validation boundary

The current patch series was checked against upstream `df462c703850fcc57443ddbcb38ebbe0d4524797`. In isolated copies, inplace conversion, repeat CMake conversion, failure guards, and source equivalence with out-of-place conversion passed. A complete wheel build and device tests have not yet been rerun for this inplace workflow.

A previous out-of-place build on `cdf531969bc07ec147adbc760be7e543b70dfdef` produced an installed wheel and passed Flash/SDPA regression tests; those results are historical, not validation of this workflow. For each new upstream SHA, repeat patch, build, and device checks before distributing the wheel.

## License

This repository uses the BSD 3-Clause License (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Copyright for newly added compatibility code and modifications is held by Zhejiang Runhe Cloud Intelligence Technology Co., Ltd. The patches derive from public [PyTorch-for-SAIL](https://github.com/t-head/pytorch-for-sail); upstream copyright and license notices are retained and are not replaced by deployment attribution.
