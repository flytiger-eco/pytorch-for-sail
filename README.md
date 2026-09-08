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

---

## Supported Hardware

- Zhenwu M890
- Zhenwu 810
- Zhenwu 810E
- Zhenwu 610
- Zhenwu 610E

---

## User Guide

To use PyTorch-for-SAIL directly via Docker or install it from PyPI, please refer to the [PyTorch-for-SAIL User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=2909).

---

## Build from Source

If you need to build and install from source, choose one of the following Docker-based workflows.

### Method 1: Build with `Dockerfile.ubuntu-ppu`

[`Dockerfile.ubuntu-ppu`](Dockerfile.ubuntu-ppu) builds the source tree into a wheel in a PPU SDK environment. The build context must be the repository root and must include the initialized `flash-attention`, `cutlass`, and `cudafy-for-sail` submodules.

```bash
# Initialize submodules before Docker sends the build context.
git submodule sync
git submodule update --init --recursive

# Build the runtime image. This also builds a wheel, installs it with runtime
# dependencies, and verifies that torch can be imported outside the source
# directory with CUDA support compiled in.
docker buildx build --load --progress=plain \
  --target runtime \
  -t pytorch-for-sail:ubuntu-ppu \
  -f Dockerfile.ubuntu-ppu .
```

The default `BASE_IMAGE` is pinned to an immutable digest. `--build-arg BASE_IMAGE=<image>` can replace the base image, but the replacement should also be digest-pinned and must provide the same SDK path and toolchain layout. The build-time validation initializes the SDK environment, checks general build tools, and fails unless the installed wheel reports CUDA support as compiled. Build options can also be overridden, for example:

```bash
docker buildx build --load \
  --build-arg MAX_JOBS=16 \
  --build-arg TORCH_CUDA_ARCH_LIST=8.9 \
  --build-arg USE_FLASH_ATTENTION=False \
  --build-arg USE_MEM_EFF_ATTENTION=False \
  --target runtime \
  -t pytorch-for-sail:ubuntu-ppu-no-attention \
  -f Dockerfile.ubuntu-ppu .
```

Start the built image with a shell:

```bash
docker run -it --rm --entrypoint /bin/bash pytorch-for-sail:ubuntu-ppu
```

The runtime image starts in `/workspace`; the source checkout remains available at `/workspace/pytorch-for-sail`. This keeps the installed wheel ahead of the source tree during normal Python imports.

For workloads that require an accelerator device, add the device mapping required by the host environment.

Verify the image independently after it is built:

```bash
docker run -t --rm --entrypoint /bin/bash pytorch-for-sail:ubuntu-ppu -lc '
  set -euo pipefail
  source /usr/local/PPU_SDK/envsetup.sh
  cd /tmp
  python3 -c "import sys, torch; compiled = torch.cuda._is_compiled(); print(torch.__version__); print(compiled); sys.exit(0 if compiled else 'torch was built without CUDA support')"
'
```

To export the generated wheel without loading the runtime image, use the `artifact` target:

```bash
rm -rf wheels
docker buildx build --progress=plain \
  --target artifact \
  --output type=local,dest=./wheels \
  -f Dockerfile.ubuntu-ppu .
```

### Method 2: Build inside a downloaded Flytiger PPU Docker image

Download a compatible PPU Docker image from the [Flytiger download page](https://www.flytiger-eco.com/download?businessType=DOCKER), then set `FLYTIGER_PPU_IMAGE` to its local image name or tag. This workflow keeps the source checkout on the host and performs the build in the container. If the download is an image archive, import it first:

```bash
docker load -i <downloaded-image-archive.tar>
```

```bash
# 1. Clone the PyTorch source code and initialize submodules
git clone --recursive https://github.com/flytiger-eco/pytorch-for-sail.git -b v2.11.0
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

# Install build dependencies
pip install -r requirements.txt

# Optional: uncomment these variables to print verbose build logs for troubleshooting
# export CUDA_VERBOSE_BUILD=1      # Print full commands used to compile CUDA sources
# export CMAKE_VERBOSE_MAKEFILE=1  # Print every compile and link command generated by CMake

# PPU toolchain-specific behavior: 8.0 enables mixed compilation for SM80 and SM89.
# Under standard CUDA semantics, 8.0 normally denotes SM80 only.
export TORCH_CUDA_ARCH_LIST="8.0"
# To build for SM89 only, comment out the line above and uncomment the following line.
# export TORCH_CUDA_ARCH_LIST="8.9"

# 4. Build the wheel package with the configured build backend
NCCL_INCLUDE_DIR=/usr/local/PPU_SDK/CUDA_SDK/include \
NCCL_LIB_DIR=/usr/local/PPU_SDK/CUDA_SDK/lib64 \
PYTORCH_VERSION=2.11.0 \
PYTORCH_BUILD_VERSION=2.11.0 \
PYTORCH_BUILD_NUMBER=0 \
USE_FLASH_ATTENTION=True \
USE_MEM_EFF_ATTENTION=True \
USE_NCCL=True \
USE_DISTRIBUTED=True \
USE_SYSTEM_NCCL=1 \
BUILD_CAFFE2=False \
BUILD_TEST=True \
python3 -m pip wheel --no-build-isolation --no-deps -w dist .

# 5. Install the built wheel package and its runtime dependencies
python3 -m pip install --force-reinstall dist/*.whl
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

For the PyTorch-for-SAIL license, please see the [LICENSE](LICENSE) file.

## Acknowledgments

PyTorch-for-SAIL is developed based on the community open-source PyTorch project. We thank the PyTorch team and the open-source community for their contributions, and welcome developers to contribute code, documentation, and tests to PyTorch-for-SAIL.
