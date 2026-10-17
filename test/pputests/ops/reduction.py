"""Reduction operators: Softmax.

Softmax forces an FP32 accumulation on the PPU and is compared against an fp64
golden.  The ``dim`` parameter layer covers ``0 / -1 / -2 / middle``; the
``MASKFILL(-inf)`` pattern covers full/partial masking (all-``-inf`` rows yield
NaN in both golden and PPU, validated by the hard NaN-consistency rule).
"""

from __future__ import annotations

import torch

from core.inputgen import (
    ALTERNATING,
    EXTREME_VALUES,
    MASKFILL,
    NORMAL,
    SORTED_ASC,
    SORTED_DESC,
    UNIFORM10,
    ZEROS,
    preset_shapes,
)
from core.opspec import BackwardMode, OpSpec, register
from core.tolerance import BF16, FP32, ToleranceOverride

_RED_SHAPES = preset_shapes("reduction")


def _norm_dim(x: torch.Tensor, dim: int) -> int:
    """Clamp ``dim`` into ``[-ndim, ndim-1]`` so 1-D shapes stay valid.

    Both golden and PPU use the same normalised dim (a pure function of ndim),
    so the comparison stays apples-to-apples even for degenerate shapes.
    """
    nd = x.dim()
    if nd == 0:
        return 0
    d = dim + nd if dim < 0 else dim
    return max(0, min(d, nd - 1))


def _softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.softmax(x, _norm_dim(x, dim))


@register
def softmax():
    return OpSpec(
        name="softmax",
        category="reduction",
        dtypes=[BF16, FP32],
        shapes=_RED_SHAPES,
        patterns=[
            NORMAL, ZEROS, UNIFORM10,
            MASKFILL(float("-inf"), 0.25),   # masked attention pattern
            MASKFILL(float("-inf"), 1.0),    # full mask -> all-NaN row
            EXTREME_VALUES(0.01),            # large-logit overflow protection
            ALTERNATING,                     # reduction-order sensitivity
            SORTED_ASC, SORTED_DESC,         # reduction stability
        ],
        params=[
            {"dim": -1},
            {"dim": -2},
            {"dim": 0},
            {"dim": 1},   # 中间维 (clamped to a valid axis for low-rank shapes)
        ],
        reffn=_softmax,
        ppufn=_softmax,
        has_backward=BackwardMode.REQUIRED,
        tolerance=ToleranceOverride(
            reduction=True,
            reason="Softmax 归约累加误差：FP32 abs 阈值放宽到 1e-4 (§4.3)。",
        ),
        notes="归约强制 FP32 累加；全 mask(-inf) 行输出 NaN，由硬性 NaN 一致性判据校验。",
    )
