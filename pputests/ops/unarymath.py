"""Unary math operators: Exp.

``exp`` has an unbounded output range (up to ~3.4e38 in FP32), so the absolute
error threshold is meaningless once outputs are large -- 1 ULP at ``exp(88)`` is
already ~1e31.  The override therefore disables the abs gate and leans on the
**relative** error, cosine similarity and the hard Inf/NaN consistency rule
(overflow ``> 88.7 -> +inf``, underflow ``< -87 -> 0``) to judge accuracy.
"""

from __future__ import annotations

import torch

from core.inputgen import (
    EXTREME_VALUES,
    MASKFILL,
    NEARBOUNDARY,
    NORMAL,
    UNIFORM10,
    ZEROS,
    Linspace,
    preset_shapes,
)
from core.opspec import BackwardMode, OpSpec, register
from core.tolerance import FP32, ToleranceOverride

_UNARY_SHAPES = preset_shapes("unary")


@register
def exp():
    return OpSpec(
        name="exp",
        category="unarymath",
        dtypes=[FP32],
        shapes=_UNARY_SHAPES,
        patterns=[
            NORMAL, ZEROS, UNIFORM10,
            Linspace(-100.0, 100.0),                    # crosses over/underflow
            NEARBOUNDARY(values=[88.7, -87.0], ulp_span=8),
            EXTREME_VALUES(0.02),                       # inf/NaN propagation
            MASKFILL(float("-inf"), 0.1),               # exp(-inf) -> 0
        ],
        params=[{}],
        reffn=lambda x: torch.exp(x),
        ppufn=lambda x: torch.exp(x),
        has_backward=BackwardMode.REQUIRED,
        tolerance=ToleranceOverride(
            max_abs_err=1e30,  # effectively disabled; rel err is the judge
            max_rel_err=1e-5,
            reason="exp 输出动态范围极大(可达 ~3e38)，绝对误差随量级失真；"
            "主判据改为相对误差 1e-5 + Inf/NaN 一致性。",
        ),
        notes="上溢 >88.7 -> +inf，下溢 <-87 -> 0；inf/0 位置与符号由硬性判据校验。",
    )
