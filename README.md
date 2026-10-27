# PyTorch-for-SAIL

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-BSD-green)](LICENSE)

[English](README.md) | [简体中文](README.zh.md)

---

## Table of Contents

- [Introduction](#introduction)
- [Supported Hardware](#supported-hardware)
- [User Guide](#user-guide)
- [Build from Source](#build-from-source)
- [Resources](#resources)
- [Security](#security)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

PyTorch-for-SAIL is developed based on the community open-source PyTorch project. It provides system-level adaptation and performance optimization for Zhenwu PPU hardware, aiming to offer developers an out-of-the-box experience for deep learning training and inference.

The project continuously tracks official PyTorch 2.x releases. While maintaining compatibility with upstream functionality, it adds backend adaptations and performance optimizations for Zhenwu PPU. The current version supports SDPA (Scaled Dot-Product Attention) backend adaptation for Zhenwu PPU, including Flash-Attention and Memory-Efficient Attention implementations.

This project is a derivative work of PyTorch. The original PyTorch copyright and license notices are retained in [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

## Supported Hardware

- Zhenwu M890
- Zhenwu 810
- Zhenwu 805
- Zhenwu 610
- Zhenwu 810E
- Zhenwu 610E

---

## User Guide

To use PyTorch-for-SAIL directly via Docker or install it from PyPI, please refer to the [PyTorch-for-SAIL User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=99).

---

## Build from Source

If you need to build and install from source, choose one of the following Docker-based workflows.

### Method 1: Build with `Dockerfile.ubuntu-ppu`

[`Dockerfile.ubuntu-ppu`](docker/ppu/Dockerfile.ubuntu-ppu) builds the source tree into a wheel inside the Flytiger PPU SDK image. It reuses the same CI scripts under [`.ci/ppu/`](.ci/ppu) that the official pipeline runs, so the PPU SDK and the v2.2 PCCL NCCL wrapper are resolved automatically. The build context must be the repository root and must include the initialized PPU submodules (`cutlass`, `flash-attention`, `xformers`, `cudafy-for-sail`, `flex-flash-attention`, and the nested `flex-flash-attention/csrc/actlize`).

```bash
# Initialize submodules before Docker sends the build context.
git submodule sync
git submodule update --init --recursive

# Build the runtime image. This also builds a wheel, installs it, and verifies that
# torch can be imported outside the source directory with CUDA support compiled in.
docker buildx build --load --progress=plain \
  --target runtime \
  -t pytorch-for-sail:ubuntu-ppu \
  -f docker/ppu/Dockerfile.ubuntu-ppu .
```

The default `BASE_IMAGE` is pinned to the PPU SDK 2.2.0 image used by CI. `--build-arg BASE_IMAGE=<image>` can replace it, but the replacement must provide the same SDK path and toolchain layout. Build options can be overridden with `--build-arg`, for example:

```bash
docker buildx build --load \
  --build-arg MAX_JOBS=16 \
  --build-arg TORCH_CUDA_ARCH_LIST=8.9 \
  --target runtime \
  -t pytorch-for-sail:ubuntu-ppu-sm89 \
  -f docker/ppu/Dockerfile.ubuntu-ppu .
```

Supported build args: `BASE_IMAGE`, `TORCH_CUDA_ARCH_LIST` (default `8.0a;8.9`, i.e. mixed SM80+SM89), `MAX_JOBS` (default `0`, auto-derived from CPU count and memory), `PPU_LINK_JOBS`, `PPU_MEM_PER_JOB_GB`, `SDK_INSTALL_DIR`, and `PIP_INDEX_URL`.

Start the built image with a shell:

```bash
docker run -it --rm --entrypoint /bin/bash pytorch-for-sail:ubuntu-ppu
```

The runtime image starts in `/workspace`; the source checkout remains available at `/workspace/pytorch-for-sail`, which keeps the installed wheel ahead of the source tree during normal Python imports. For workloads that require an accelerator device, add the device mapping required by the host environment.

Verify the image independently after it is built:

```bash
docker run -t --rm --entrypoint /bin/bash pytorch-for-sail:ubuntu-ppu -lc '
  set -euo pipefail
  source /usr/local/PPU_SDK/envsetup.sh
  cd /tmp
  python3 -c "import torch; print(torch.__version__); assert torch.cuda._is_compiled(), \"torch was built without CUDA support\""
'
```

To export the generated wheel without loading the runtime image, use the `artifact` target:

```bash
rm -rf wheels
docker buildx build --progress=plain \
  --target artifact \
  --output type=local,dest=./wheels \
  -f docker/ppu/Dockerfile.ubuntu-ppu .
```

### Method 2: Build with `Dockerfile.sdk2.2-ppu` (SDK v2.2 + decoupled PCCL/NCCL)

[`Dockerfile.sdk2.2-ppu`](docker/ppu/Dockerfile.sdk2.2-ppu) targets PPU SDK v2.2 (`sdk2.2.0-cuda13.0-ubuntu24-py312`). Starting from SDK v2.2, PCCL and its NCCL compatibility wrapper are shipped as a package decoupled from the SDK, so this Dockerfile installs that package explicitly before compiling. The build context must be the repository root and must include the initialized `flash-attention`, `flex-flash-attention`, `cutlass`, `cudafy-for-sail`, and `xformers` submodules.

```bash
# Initialize submodules before Docker sends the build context.
git submodule sync
git submodule update --init --recursive

# Build the runtime image. This also builds a wheel, installs it with runtime
# dependencies, and verifies that torch can be imported outside the source
# directory with CUDA support compiled in.
docker buildx build --load --progress=plain \
  --target runtime \
  -t pytorch-for-sail:2.13-sdk2.2-ppu \
  -f docker/ppu/Dockerfile.sdk2.2-ppu .
```

The default `BASE_IMAGE` is pinned to an immutable digest. The decoupled PCCL/NCCL package is controlled by two build arguments:

- `NCCL_PACKAGE_URL`: the PCCL/NCCL tarball matching the artifact version, CUDA version, and OS. The default points at the `v2.2.0` / `nccl-2.27.3` / `ubuntu2404` / `cuda13.0` package. Download it manually from the [Flytiger NCCL download page](https://www.flytiger-eco.com/download?businessType=NCCL) if you need a different build.
- `NCCL_PACKAGE_SHA256`: the expected checksum of that tarball. The build verifies it after download.

The tarball extracts a top-level `pccl/` directory into `/usr/local` (-> `/usr/local/pccl`), providing both the native PCCL layer and the NCCL compatibility wrapper (`/usr/local/pccl/nccl_wrapper`). If the URL is unreachable from the build network (for example, an authenticated artifact registry that returns `403` inside the container), the build falls back to the identical copy pre-installed in the base image, then still validates the required SDK, PCCL, and NCCL artifacts. Override either argument as needed:

```bash
docker buildx build --load \
  --build-arg MAX_JOBS=16 \
  --build-arg NCCL_PACKAGE_URL=<local-or-mirror-url>/nccl-2.27.3-v13_ubuntu2404_sdk-v2.2.0.tar.gz \
  --build-arg NCCL_PACKAGE_SHA256=<expected-sha256> \
  --target runtime \
  -t pytorch-for-sail:2.13-sdk2.2-ppu \
  -f docker/ppu/Dockerfile.sdk2.2-ppu .
```

Start the built image with a shell (the base image ships its own `ENTRYPOINT`, so override it):

```bash
docker run -it --rm --entrypoint /bin/bash pytorch-for-sail:2.13-sdk2.2-ppu
```

The runtime image starts in `/workspace`; the source checkout remains available at `/workspace/pytorch-for-sail`. For workloads that require an accelerator device, add the device mapping required by the host environment.

Verify the image independently after it is built:

```bash
docker run --rm -i --entrypoint /bin/bash pytorch-for-sail:2.13-sdk2.2-ppu <<'EOF'
set -euo pipefail
source /usr/local/PPU_SDK/envsetup.sh
source /usr/local/pccl/envsetup.sh
cd /tmp
python3 -c "import sys, torch; compiled = torch.cuda._is_compiled(); print(torch.__version__); print(compiled); print(torch.cuda.nccl.version()); sys.exit(0 if compiled else 1)"
EOF
```

To export the generated wheel without loading the runtime image, use the `artifact` target:

```bash
rm -rf wheels
docker buildx build --progress=plain \
  --target artifact \
  --output type=local,dest=./wheels \
  -f docker/ppu/Dockerfile.sdk2.2-ppu .
```

### Method 3: Build inside a downloaded Flytiger PPU Docker image

Download a compatible PPU Docker image from the [Flytiger download page](https://www.flytiger-eco.com/download?businessType=DOCKER), then set `FLYTIGER_PPU_IMAGE` to its local image name or tag. This workflow keeps the source checkout on the host and performs the build inside the container. If the download is an image archive, import it first:

```bash
docker load -i <downloaded-image-archive.tar>
```

```bash
# 1. Clone the PyTorch source code and initialize submodules
git clone --recursive https://github.com/flytiger-eco/pytorch-for-sail.git -b v2.13.0
cd pytorch-for-sail

# If --recursive was not used or submodules are incomplete, run:
git submodule sync
git submodule update --init --recursive

# 2. Start the downloaded PPU image with the source checkout mounted.
# Replace the placeholder with the local image name or tag obtained above.
export FLYTIGER_PPU_IMAGE=<downloaded-flytiger-ppu-image>
docker run -it --rm \
  -v "$(pwd):/workspace/pytorch-for-sail" \
  -w /workspace/pytorch-for-sail \
  --entrypoint /bin/bash \
  "${FLYTIGER_PPU_IMAGE}"

# Run the following commands inside the PyTorch-for-SAIL Docker container
# 3. Configure the build environment
source /usr/local/PPU_SDK/envsetup.sh

# The default configuration applies to artifact version v2.2, which requires
# the separately provided PCCL NCCL wrapper. Download it from
# https://www.flytiger-eco.com/download?businessType=NCCL and select a package
# matching the artifact version, CUDA version, and operating system. Extract it
# only once per container.
tar zxf <matching-nccl-wrapper-package>.tar.gz -C /usr/local
source /usr/local/pccl/envsetup.sh
export NCCL_INCLUDE_DIR="${NCCL_HOME}/include"
export NCCL_LIB_DIR="${NCCL_HOME}/lib"

# Artifact version v2.1 provides NCCL in the SDK and does not require the wrapper.
# Replace the four lines above with the following SDK configuration for v2.1.
# export NCCL_INCLUDE_DIR="${CUDA_SDK}/include"
# export NCCL_LIB_DIR="${CUDA_SDK}/lib64"

# Install build dependencies
pip install -r requirements.txt

# Optional: uncomment these variables to print verbose build logs for troubleshooting
# export CUDA_VERBOSE_BUILD=1      # Print full commands used to compile CUDA sources
# export CMAKE_VERBOSE_MAKEFILE=1  # Print every compile and link command generated by CMake

# Build architecture: the Docker builds (Methods 1-2) default to "8.0a;8.9"; export the
# same value here so the manual build matches. TORCH_CUDA_ARCH_LIST="8.0a;8.9" compiles
# kernels for the SM80 family and SM89, so one wheel runs on every supported Zhenwu board
# (610/610E/810/810E and M890). PPU toolchain note: unlike standard CUDA, where 8.0 means
# SM80 only, here 8.0/8.0a enables mixed SM80+SM89 compilation; if this variable is left
# unset, setup.py falls back to "8.0". To target one architecture, override it (e.g. "8.9"
# for SM89 / Zhenwu M890 only).
export TORCH_CUDA_ARCH_LIST="8.0a;8.9"

# 3. Optional build options
# By default, the elementwise operator optimization and the Flex Flash Attention backend are
# compiled into the wheel. Both features are disabled by default at runtime and can be enabled
# only on Zhenwu M890; see step 6 for runtime switches.
# To exclude the elementwise operator optimization from the build, set the following environment
# variable. Once excluded, it cannot be enabled at runtime:
# export USE_ELEMENTWISE_OPT=False
# To exclude the Flex Flash Attention backend from the build, set the following environment
# variable. Once excluded, it cannot be enabled at runtime:
# export USE_FLEX_FLASH_ATTENTION=False

# 4. Build the wheel package. NCCL_INCLUDE_DIR and NCCL_LIB_DIR are exported
# by the selected artifact-version configuration above.
PYTORCH_VERSION=2.13.0 \
PYTORCH_BUILD_VERSION=2.13.0 \
PYTORCH_BUILD_NUMBER=0 \
USE_FLASH_ATTENTION=True \
USE_MEM_EFF_ATTENTION=True \
USE_NCCL=True \
USE_DISTRIBUTED=True \
USE_SYSTEM_NCCL=1 \
BUILD_CAFFE2=False \
BUILD_TEST=True \
python3 setup.py bdist_wheel

# 5. Install the built wheel package
pip install dist/*.whl

# 6. Optional runtime switches
# The elementwise operator optimization accelerates selected special memory access patterns.
# It is disabled by default on Zhenwu M890 machines and always disabled on Zhenwu 810E machines,
# where it cannot be enabled through the environment variable.
# To enable it on Zhenwu M890, set the following variable before starting the Python process;
# setting it after `import torch` has no effect.
# export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1
# Flex Flash Attention is experimental, disabled by default, and available only on Zhenwu M890.
# To enable it, set the following variable before starting the Python process.
# Setting a variable has no effect if the feature was disabled at build time.
# export PYTORCH_ENABLE_PPU_FLEX_FLASH_ATTENTION=1
```

---

## Resources

- [PyTorch Official Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## Security

For security information, please see [SECURITY.md](SECURITY.md).

## Disclaimer

- This software is provided for development and debugging purposes. Users assume all risks associated with its use.
- Users are responsible for managing data generated during use and complying with applicable security and compliance requirements.

## License

For the PyTorch-for-SAIL license, please see the [LICENSE](LICENSE) file. The upstream PyTorch copyright and third-party attribution notices are retained in [NOTICE](NOTICE).

## Acknowledgments

PyTorch-for-SAIL is developed based on the community open-source PyTorch project. We thank the PyTorch team and the open-source community for their contributions, and welcome developers to contribute code, documentation, and tests to PyTorch-for-SAIL.
