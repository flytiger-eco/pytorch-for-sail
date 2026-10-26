# PyTorch-for-SAIL

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-BSD-green)](LICENSE)

[English](README.md) | [简体中文](README.zh.md)

---

## 目录

- [简介](#简介)
- [支持的硬件型号](#支持的硬件型号)
- [用户指南](#用户指南)
- [源码编译](#源码编译)
- [资源链接](#资源链接)
- [安全声明](#安全声明)
- [免责声明](#免责声明)
- [许可证](#许可证)
- [致谢](#致谢)

---

## 简介

PyTorch-for-SAIL 基于社区开源 PyTorch 项目开发，面向真武 PPU 硬件进行系统级适配与性能优化，旨在为开发者提供开箱即用的深度学习训练与推理能力。

项目持续跟进 PyTorch 2.x 官方版本，在保持上游功能兼容的基础上，补充真武 PPU 相关后端适配和性能优化。当前版本支持 SDPA（Scaled Dot-Product Attention）模块的真武 PPU 后端适配，包括 Flash-Attention 与 Memory-Efficient Attention 实现。

本项目是 PyTorch 的衍生作品。PyTorch 原始的版权与许可声明保留在 [LICENSE](LICENSE) 与 [NOTICE](NOTICE) 中。

---

## 支持的硬件型号

- 真武 M890
- 真武 810
- 真武 805
- 真武 610
- 真武 810E
- 真武 610E

---

## 用户指南

如需直接通过 Docker 使用 PyTorch-for-SAIL 或从 PyPI 安装，请参阅 [PyTorch-for-SAIL 用户指南](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=99)。

---

## 源码编译

如需从源码编译并安装，可选择以下两种基于 Docker 的工作流。

### 方式一：使用 `Dockerfile.ubuntu-ppu` 构建

[`Dockerfile.ubuntu-ppu`](Dockerfile.ubuntu-ppu) 会在 Flytiger PPU SDK 镜像中将源码编译为 wheel。它复用官方流水线所使用的 [`.ci/ppu/`](.ci/ppu) 下的 CI 脚本，因此 PPU SDK 与 v2.2 的 PCCL NCCL wrapper 都会被自动解析。Docker 构建上下文必须是仓库根目录，并且必须包含已初始化的 PPU 子模块（`cutlass`、`flash-attention`、`xformers`、`cudafy-for-sail`、`flex-flash-attention`，以及嵌套的 `flex-flash-attention/csrc/actlize`）。

```bash
# Docker 发送构建上下文前，先初始化子模块。
git submodule sync
git submodule update --init --recursive

# 构建运行时镜像。该过程还会编译 wheel、安装它，并在源码目录之外验证
# torch 可导入且已编译 CUDA 支持。
docker buildx build --load --progress=plain \
  --target runtime \
  -t pytorch-for-sail:ubuntu-ppu \
  -f Dockerfile.ubuntu-ppu .
```

默认 `BASE_IMAGE` 已固定为 CI 使用的 PPU SDK 2.2.0 镜像。可通过 `--build-arg BASE_IMAGE=<image>` 替换基础镜像，但替换后的镜像必须提供相同的 SDK 路径和工具链布局。也可通过 `--build-arg` 覆盖编译选项，例如：

```bash
docker buildx build --load \
  --build-arg MAX_JOBS=16 \
  --build-arg TORCH_CUDA_ARCH_LIST=8.9 \
  --target runtime \
  -t pytorch-for-sail:ubuntu-ppu-sm89 \
  -f Dockerfile.ubuntu-ppu .
```

支持的构建参数：`BASE_IMAGE`、`TORCH_CUDA_ARCH_LIST`（默认 `8.0a;8.9`，即 SM80+SM89 混合编译）、`MAX_JOBS`（默认 `0`，按 CPU 核数与内存自动推导）、`PPU_LINK_JOBS`、`PPU_MEM_PER_JOB_GB`、`SDK_INSTALL_DIR`、`PIP_INDEX_URL`。

以 Shell 启动构建完成的镜像：

```bash
docker run -it --rm --entrypoint /bin/bash pytorch-for-sail:ubuntu-ppu
```

运行时镜像默认从 `/workspace` 启动；源码检出仍位于 `/workspace/pytorch-for-sail`，这可确保常规 Python 导入优先使用已安装 wheel，而不是源码树。需要使用加速设备的工作负载，应按宿主机环境要求添加设备映射。

镜像构建完成后，可独立执行以下冒烟验证：

```bash
docker run -t --rm --entrypoint /bin/bash pytorch-for-sail:ubuntu-ppu -lc '
  set -euo pipefail
  source /usr/local/PPU_SDK/envsetup.sh
  cd /tmp
  python3 -c "import torch; print(torch.__version__); assert torch.cuda._is_compiled(), \"torch was built without CUDA support\""
'
```

如需仅导出生成的 wheel 而不加载运行时镜像，可使用 `artifact` target：

```bash
rm -rf wheels
docker buildx build --progress=plain \
  --target artifact \
  --output type=local,dest=./wheels \
  -f Dockerfile.ubuntu-ppu .
```

### 方式二：下载 Flytiger PPU Docker 镜像后自行编译

从 [Flytiger 下载页面](https://www.flytiger-eco.com/download?businessType=DOCKER) 下载兼容的 PPU Docker 镜像，并将 `FLYTIGER_PPU_IMAGE` 设置为本地镜像名称或 tag。该方式保留宿主机源码检出，并在容器内完成编译。若下载内容为镜像归档文件，先执行导入：

```bash
docker load -i <downloaded-image-archive.tar>
```

```bash
# 1. 下载 PyTorch 源码并初始化子模块
git clone --recursive https://github.com/flytiger-eco/pytorch-for-sail.git -b v2.13.0
cd pytorch-for-sail

# 如果 clone 时未使用 --recursive，或子模块拉取不完整，请执行：
git submodule sync
git submodule update --init --recursive

# 2. 挂载源码并启动下载的 PPU 镜像。
# 将占位符替换为上一步获取的本地镜像名称或 tag。
export FLYTIGER_PPU_IMAGE=<downloaded-flytiger-ppu-image>
docker run -it --rm \
  -v "$(pwd):/workspace/pytorch-for-sail" \
  -w /workspace/pytorch-for-sail \
  --entrypoint /bin/bash \
  "${FLYTIGER_PPU_IMAGE}"

# 以下命令需在 PyTorch-for-SAIL Docker 容器内执行
# 3. 配置编译环境
source /usr/local/PPU_SDK/envsetup.sh

# 默认配置适用于制品版本 v2.2：需要单独安装 PCCL 提供的 NCCL wrapper。
# 从 https://www.flytiger-eco.com/download?businessType=NCCL 下载 wrapper 安装包，
# 并按制品版本、CUDA 版本和操作系统选择匹配的包；每个容器只需解压一次。
tar zxf <匹配的-nccl-wrapper-安装包>.tar.gz -C /usr/local
source /usr/local/pccl/envsetup.sh
export NCCL_INCLUDE_DIR="${NCCL_HOME}/include"
export NCCL_LIB_DIR="${NCCL_HOME}/lib"

# 制品版本 v2.1：NCCL 随 SDK 提供，不需要安装 wrapper；改用下方 SDK 配置替换上方四行。
# export NCCL_INCLUDE_DIR="${CUDA_SDK}/include"
# export NCCL_LIB_DIR="${CUDA_SDK}/lib64"

# 安装编译依赖
pip install -r requirements.txt

# 可选：取消以下变量的注释，以输出用于排查问题的详细编译日志
# export CUDA_VERBOSE_BUILD=1      # 打印编译 CUDA 源文件的完整命令行
# export CMAKE_VERBOSE_MAKEFILE=1  # 打印 CMake 生成的每条编译和链接命令

# PPU 工具链特有行为：8.0a 仅表示 SM80，8.9 仅表示 SM89。
# 当需要启动 SM80 和 SM89 混合编译时，可以使用如下表达方式：
export TORCH_CUDA_ARCH_LIST="8.0a;8.9"
# 如需仅编译 SM89，请注释上一行并取消下一行的注释。
# export TORCH_CUDA_ARCH_LIST="8.9"

# 3. 可选编译选项
# 默认会将 elementwise 算子优化和 Flex Flash Attention 后端编译进 wheel 包；
# 两项功能在运行时均默认关闭，仅可在真武 M890 上开启；开关方式见第 6 步。
# 如需在编译时去掉 elementwise 算子优化，请设置以下环境变量；
# export USE_ELEMENTWISE_OPT=False
# 如需在编译时去掉 Flex Flash Attention 后端，请设置以下环境变量；
# export USE_FLEX_FLASH_ATTENTION=False

# 4. 编译生成 wheel 安装包。NCCL_INCLUDE_DIR 和 NCCL_LIB_DIR 已由上方版本配置导出。
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

# 5. 安装编译生成的 wheel 包
pip install dist/*.whl

# 6. 可选运行时开关
# elementwise 算子优化针对部分特殊访存模式提供性能加速，仅在真武 M890 机器上生效并默认关闭。
# 真武 810E 机器上始终关闭，无法通过环境变量开启。
# 如需在真武 M890 机器上开启该优化，须在启动 Python 进程前设置以下环境变量；
# 在 Python 中 import torch 之后再设置不会生效。
# export PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=1
# Flex Flash Attention 为实验特性，默认关闭，仅在真武 M890 机器上生效；
# 如需开启，须在启动 Python 进程前设置以下环境变量。
# 编译时已关闭的功能，设置对应变量也不会生效。
# export PYTORCH_ENABLE_PPU_FLEX_FLASH_ATTENTION=1
```

---

## 资源链接

- [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/index.html)
- [PyTorch 教程](https://pytorch.org/tutorials/)

---

## 安全声明

安全相关说明请参见 [SECURITY.md](SECURITY.md)。

## 免责声明

- 本软件仅供开发和调试使用，使用者需自行承担使用风险。
- 用户需自行管理运行过程中产生的数据，并遵守相关安全和合规要求。

## 许可证

PyTorch-for-SAIL 的使用许可证，请参见 [LICENSE](LICENSE) 文件。上游 PyTorch 的版权与第三方归属声明保留在 [NOTICE](NOTICE) 中。

## 致谢

PyTorch-for-SAIL 基于社区开源 PyTorch 项目开发。感谢 PyTorch 团队和开源社区的贡献，欢迎开发者参与 PyTorch-for-SAIL 的代码、文档和测试贡献。
