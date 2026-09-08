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

---

## 支持的硬件型号

- 真武 M890
- 真武 810
- 真武 810E
- 真武 610
- 真武 610E

---

## 用户指南

如需直接通过 Docker 使用 PyTorch-for-SAIL 或从 PyPI 安装，请参阅 [PyTorch-for-SAIL 用户指南](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=2909)。

---

## 源码编译

如需从源码编译并安装，可选择以下两种基于 Docker 的工作流。

### 方式一：使用 `Dockerfile.ubuntu-ppu` 构建

[`Dockerfile.ubuntu-ppu`](Dockerfile.ubuntu-ppu) 会在 PPU SDK 环境中将源码编译为 wheel。Docker 构建上下文必须是仓库根目录，并且必须包含已初始化的 `flash-attention`、`cutlass` 和 `cudafy-for-sail` 子模块。

```bash
# Docker 发送构建上下文前，先初始化子模块。
git submodule sync
git submodule update --init --recursive

# 构建运行时镜像。该过程还会编译 wheel、安装其运行时依赖，并在源码目录外验证
# torch 可导入且已编译 CUDA 支持。
docker buildx build --load --progress=plain \
  --target runtime \
  -t pytorch-for-sail:ubuntu-ppu \
  -f Dockerfile.ubuntu-ppu .
```

默认 `BASE_IMAGE` 已固定为不可变 digest。可通过 `--build-arg BASE_IMAGE=<image>` 替换基础镜像，但替换后的镜像也建议固定 digest，并且必须满足相同的 SDK 路径和工具链布局。构建期自检会初始化 SDK 环境、检查通用构建工具，并且当已安装 wheel 未报告编译 CUDA 支持时失败。也可覆盖编译选项，例如：

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

以 Shell 启动构建完成的镜像：

```bash
docker run -it --rm --entrypoint /bin/bash pytorch-for-sail:ubuntu-ppu
```

运行时镜像默认从 `/workspace` 启动；源码检出仍位于 `/workspace/pytorch-for-sail`。这可确保常规 Python 导入优先使用已安装 wheel，而不是源码树。

需要使用加速设备的工作负载，应按宿主机环境要求添加设备映射。

镜像构建完成后，可独立执行以下冒烟验证：

```bash
docker run -t --rm --entrypoint /bin/bash pytorch-for-sail:ubuntu-ppu -lc '
  set -euo pipefail
  source /usr/local/PPU_SDK/envsetup.sh
  cd /tmp
  python3 -c "import sys, torch; compiled = torch.cuda._is_compiled(); print(torch.__version__); print(compiled); sys.exit(0 if compiled else 'torch was built without CUDA support')"
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
git clone --recursive https://github.com/flytiger-eco/pytorch-for-sail.git -b v2.11.0
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

# 安装编译依赖
pip install -r requirements.txt

# 可选：取消以下变量的注释，以输出用于排查问题的详细编译日志
# export CUDA_VERBOSE_BUILD=1      # 打印编译 CUDA 源文件的完整命令行
# export CMAKE_VERBOSE_MAKEFILE=1  # 打印 CMake 生成的每条编译和链接命令

# PPU 工具链特有行为：8.0 会启用 SM80 和 SM89 混合编译。
# 在标准 CUDA 语义中，8.0 通常仅表示 SM80。
export TORCH_CUDA_ARCH_LIST="8.0"
# 如需仅编译 SM89，请注释上一行并取消下一行的注释。
# export TORCH_CUDA_ARCH_LIST="8.9"

# 4. 使用配置的构建后端编译生成 wheel 安装包
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

# 5. 安装编译生成的 wheel 包及其运行时依赖
python3 -m pip install --force-reinstall dist/*.whl
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

PyTorch-for-SAIL 的使用许可证，请参见 [LICENSE](LICENSE) 文件。

## 致谢

PyTorch-for-SAIL 基于社区开源 PyTorch 项目开发。感谢 PyTorch 团队和开源社区的贡献，欢迎开发者参与 PyTorch-for-SAIL 的代码、文档和测试贡献。
