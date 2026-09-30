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
- [Resources](#resources)
- [Security](#security)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Introduction

PyTorch-for-SAIL is developed based on the community open-source PyTorch project. It provides system-level adaptation and performance optimization for Zhenwu PPU hardware, aiming to offer developers an out-of-the-box experience for deep learning training and inference.

The project continuously tracks official PyTorch 2.x releases. While maintaining compatibility with upstream functionality, it adds backend adaptations and performance optimizations for Zhenwu PPU. The current version supports SDPA (Scaled Dot-Product Attention) backend adaptation for Zhenwu PPU, including Flash-Attention and Memory-Efficient Attention implementations.

This project is a derivative work of PyTorch. The original PyTorch copyright and license notices are retained in [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

## Supported Hardware

- Zhenwu M890
- Zhenwu 810
- Zhenwu 810E
- Zhenwu 610
- Zhenwu 610E

---

## User Guide

Please refer to the PyTorch-for-SAIL User Guide.

> **Elementwise optimization**: Compiled by default (effective only for the 8.9 architecture). To disable at build time, set `USE_ELEMENTWISE_OPT=False`. At runtime, the optimization is off by default; set `PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT=True` to enable it.

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
