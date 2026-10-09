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

如需从源码编译并安装，请确保在 [PyTorch-for-SAIL Docker 镜像](https://www.flytiger-eco.com/download?businessType=DOCKER)内进行编译。

```bash
# 1. 下载 PyTorch 源码并初始化子模块
git clone --recursive https://github.com/flytiger-eco/pytorch-for-sail.git -b v2.13.0
cd pytorch-for-sail

# 如果 clone 时未使用 --recursive，或子模块拉取不完整，请执行：
git submodule sync
git submodule update --init --recursive

# 以下命令需在 PyTorch-for-SAIL Docker 容器内执行
# 2. 配置编译环境
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

# 编译期：elementwise 算子优化默认编译，该优化仅对 8.9 架构有效；如不想编译该优化，请取消下一行的注释。
# 制品版本 v2.2 默认开启以下编译选项：
# - USE_ELEMENTWISE_OPT：elementwise 算子优化，仅对 8.9 架构有效。
# - USE_FLEX_FLASH_ATTENTION：Flex Flash Attention 编译支持。
# 如不希望编译任一功能，请在执行下方构建命令前显式关闭：
# export USE_ELEMENTWISE_OPT=False
# export USE_FLEX_FLASH_ATTENTION=False
# 运行时：elementwise 算子优化默认关闭，请在运行时设置 PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=True 开启优化。
# Flex Flash Attention 后端也默认关闭；仅适用于 890P，在启动 Python 进程前设置：
# export TORCH_FLEX_FLASH_SDPA_ENABLED=1

# 3. 编译生成 wheel 安装包。NCCL_INCLUDE_DIR 和 NCCL_LIB_DIR 已由上方版本配置导出。
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

# 4. 安装编译生成的 wheel 包
pip install dist/*.whl
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
