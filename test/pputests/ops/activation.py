"""Activation operators: ReLU, ReLU6, Sigmoid, Softplus and Tanh.

Each operator is described by a single ``OpSpec`` (~15 lines).  ``reffn`` is the
fp64 CPU golden; ``ppufn`` is the PPU-side call (torch dispatched on the PPU
device).  Adding an operator means adding one ``@register`` factory below.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from core.inputgen import (
    EXTREME_VALUES,
    NEARBOUNDARY,
    NORMAL,
    ONES,
    UNIFORM10,
    ZEROS,
    FULL,
    Linspace,
    preset_shapes,
)
from core.opspec import BackwardMode, OpSpec, register
from core.tolerance import BF16, FP16, FP32

_ACT_SHAPES = preset_shapes("activation")


@register
def relu():
    return OpSpec(
        name="relu",
        category="activation",
        dtypes=[BF16, FP16, FP32],
        shapes=_ACT_SHAPES,
        patterns=[
            NORMAL, ZEROS, ONES, UNIFORM10,
            NEARBOUNDARY(values=[0.0], ulp_span=2),
            EXTREME_VALUES(0.01),
        ],
        params=[{}],
        reffn=lambda x: torch.relu(x),
        ppufn=lambda x: torch.relu(x),
        has_backward=BackwardMode.REQUIRED,
        notes="x=0 处梯度取 torch 语义 (次梯度 = 0)。",
    )


@register
def relu6():
    return OpSpec(
        name="relu6",
        category="activation",
        dtypes=[BF16, FP32],
        shapes=_ACT_SHAPES,
        patterns=[
            NORMAL, ZEROS, ONES, UNIFORM10,
            FULL(6.0),                             # constant at the upper corner
            NEARBOUNDARY(values=[0.0, 6.0], ulp_span=2),
            Linspace(5.99, 6.01),
            EXTREME_VALUES(0.01),
        ],
        params=[{}],
        reffn=lambda x: F.relu6(x),
        ppufn=lambda x: F.relu6(x),
        has_backward=BackwardMode.REQUIRED,
        notes="x=0/6 双拐点梯度取 torch hardtanh 语义 (拐点处为 0)。",
    )


@register
def sigmoid():
    return OpSpec(
        name="sigmoid",
        category="activation",
        dtypes=[BF16, FP16, FP32],
        shapes=_ACT_SHAPES,
        patterns=[
            NORMAL, ZEROS, UNIFORM10,
            NEARBOUNDARY(values=[-20.0, 20.0], ulp_span=4),
            Linspace(-20.0, 20.0),
            EXTREME_VALUES(0.01),
        ],
        params=[{}],
        reffn=lambda x: torch.sigmoid(x),
        ppufn=lambda x: torch.sigmoid(x),
        has_backward=BackwardMode.REQUIRED,
        notes="饱和区 ±20：输出趋近 0/1，梯度趋近 0，需绝对误差判据。",
    )


@register
def softplus():
    return OpSpec(
        name="softplus",
        category="activation",
        dtypes=[FP32],  # BF16 是否测待确认，本期仅 FP32
        shapes=_ACT_SHAPES,
        patterns=[
            NORMAL, ZEROS, UNIFORM10,
            Linspace(-30.0, 30.0),                 # sweep through threshold
            NEARBOUNDARY(values=[20.0], ulp_span=4),
            EXTREME_VALUES(0.01),                  # large-value linear branch
        ],
        params=[
            {"beta": 1.0, "threshold": 20},
            {"beta": 2.0, "threshold": 20},
            {"beta": 0.5, "threshold": 20},
            {"beta": 1.0, "threshold": 0},
            {"beta": 2.0, "threshold": 0},
            {"beta": 0.5, "threshold": 0},
        ],
        reffn=lambda x, beta, threshold: F.softplus(
            x, beta=beta, threshold=threshold
        ),
        ppufn=lambda x, beta, threshold: F.softplus(
            x, beta=beta, threshold=threshold
        ),
        has_backward=BackwardMode.OPTIONAL,  # 参考侧可导，是否必测待确认
        notes="beta/threshold 全组合；x*beta>threshold 走线性分支 (out=x)。",
    )


@register
def tanh():
    return OpSpec(
        name="tanh",
        category="activation",
        dtypes=[BF16, FP16, FP32],
        shapes=_ACT_SHAPES,
        patterns=[
            NORMAL, ZEROS, UNIFORM10,
            NEARBOUNDARY(values=[-10.0, 10.0], ulp_span=4),
            Linspace(-10.0, 10.0),
            EXTREME_VALUES(0.01),
        ],
        params=[{}],
        reffn=lambda x: torch.tanh(x),
        ppufn=lambda x: torch.tanh(x),
        has_backward=BackwardMode.REQUIRED,
        notes="饱和区 ±10：输出趋近 ±1，梯度趋近 0。",
    )
