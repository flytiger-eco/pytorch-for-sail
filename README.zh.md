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
- [资源链接](#资源链接)
- [安全声明](#安全声明)
- [免责声明](#免责声明)
- [许可证](#许可证)

---

## 简介

PyTorch-for-SAIL 基于社区开源 PyTorch 项目开发，面向真武 PPU 硬件进行系统级适配与性能优化，旨在为开发者提供开箱即用的深度学习训练与推理能力。

项目持续跟进 PyTorch 2.x 官方版本，在保持上游功能兼容的基础上，补充真武 PPU 相关后端适配和性能优化。当前版本支持 SDPA（Scaled Dot-Product Attention）模块的真武 PPU 后端适配，包括 Flash-Attention 与 Memory-Efficient Attention 实现。

本项目是 PyTorch 的衍生作品。PyTorch 原始的版权与许可声明保留在 [LICENSE](LICENSE) 与 [NOTICE](NOTICE) 中。

---

## 支持的硬件型号

- 真武 M890
- 真武 810
- 真武 810E
- 真武 610
- 真武 610E

---

## 用户指南

请参考 PyTorch-for-SAIL 用户指南。

> **Elementwise 算子优化**：编译期默认开启（仅对 8.9 架构有效），如不想编译该优化请设置 `USE_ELEMENTWISE_OPT=False`。运行时默认关闭，设置 `PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=True` 可开启。
>
> **Flex Flash Attention 后端**：编译期默认开启，如需关闭请设置 `USE_FLEX_FLASH_ATTENTION=False`。运行时默认关闭，在 M890 机器启动 Python 进程前设置 `TORCH_FLEX_FLASH_SDPA_ENABLED=1` 可开启；该变量不适用于 810E。

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
