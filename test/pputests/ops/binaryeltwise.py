"""Binary element-wise operators: Add, Sub and Mul.

Broadcast combinations (§5.3) are expressed through the reserved ``_shape_b``
parameter key, which the input builder consumes to size the second operand and
which is stripped before the op is called.  Because BF16 has only an 8-bit
mantissa, absolute error grows with operand magnitude (1 ULP at ~2000 is ~16),
so these specs relax the abs gate and rely on relative error + cosine + the hard
Inf/NaN rules -- each relaxation is declared with a reason per §4.3.
"""

from __future__ import annotations

import torch

from core.inputgen import (
    CANCELLATION,
    EXTREME_VALUES,
    MASKFILL,
    NORMAL,
    PatternKind,
    UNIFORM1,
    UNIFORM10,
    UNIFORM1000,
    ZEROS,
    effective_shape,
    preset_shapes,
)
from core.opspec import BackwardMode, OpSpec, register
from core.tolerance import BF16, FP16, FP32, ToleranceOverride

_BIN_SHAPES = preset_shapes("binary")

# broadcast combos applied to the second operand (§5.3)
_BCAST_PARAMS = [
    {},                          # same shape (pure element-wise)
    {"_shape_b": ()},            # scalar operand
    {"_shape_b": (1,)},          # 1-element broadcast vector
    {"_shape_b": (1, 4, 1, 1)},  # per-channel NCHW bias/gain
]

_GRAD_PATTERNS = [NORMAL, ZEROS, EXTREME_VALUES(0.01)]


def _valid_broadcast_case(shape_case, pattern, params):
    """Keep only shape products accepted by element-wise broadcasting."""
    shape_a = effective_shape(tuple(shape_case.shape), shape_case.view)
    raw_b = tuple(params.get("_shape_b", shape_case.shape))
    default_view_b = shape_case.view if "_shape_b" not in params else None
    view_b = params.get("_view_b", default_view_b)
    shape_b = effective_shape(raw_b, view_b)

    # The dedicated per-channel case is meaningful only for NCHW tensors with
    # C=4; accepting rank promotion on vectors/matrices would be technically
    # broadcastable but would no longer test the documented scenario.
    if raw_b == (1, 4, 1, 1) and (len(shape_a) != 4 or shape_a[1] != 4):
        return False

    # A cancellation pair is defined as b=a+delta and therefore requires
    # identical logical shapes; silently replacing a requested broadcast input
    # with a full-sized tensor would make the case metadata incorrect.
    if pattern.kind is PatternKind.CANCELLATION and shape_a != shape_b:
        return False
    try:
        torch.broadcast_shapes(shape_a, shape_b)
    except RuntimeError:
        return False
    return True


# abs gate relaxed to cover BF16 ULP at the uniform1000 magnitude range.
_ABS_RELAX = ToleranceOverride(
    max_abs_err=64.0,
    reason="元素级 add/sub 在 uniform1000 大幅值下，BF16 1-ULP 绝对误差随量级增大"
    "(~2000 处约 16)，绝对阈值放宽至 64；相对误差(FP32 1e-5 / BF16 1.6e-2)为主判据。",
)


@register
def add():
    return OpSpec(
        name="add",
        category="binaryeltwise",
        dtypes=[BF16, FP16, FP32],
        shapes=_BIN_SHAPES,
        patterns=[NORMAL, ZEROS, UNIFORM10, UNIFORM1000, EXTREME_VALUES(0.01)],
        params=_BCAST_PARAMS,
        reffn=lambda a, b: torch.add(a, b),
        ppufn=lambda a, b: torch.add(a, b),
        has_backward=BackwardMode.REQUIRED,
        tolerance=_ABS_RELAX,
        grad_patterns=_GRAD_PATTERNS,
        n_inputs=2,
        case_filter=_valid_broadcast_case,
        notes="覆盖 broadcast：标量、1-元素向量、per-channel (1,C,1,1)。",
    )


@register
def sub():
    return OpSpec(
        name="sub",
        category="binaryeltwise",
        dtypes=[BF16, FP32],
        shapes=_BIN_SHAPES,
        patterns=[NORMAL, UNIFORM10, CANCELLATION, EXTREME_VALUES(0.01)],
        params=_BCAST_PARAMS,
        reffn=lambda a, b: torch.sub(a, b),
        ppufn=lambda a, b: torch.sub(a, b),
        has_backward=BackwardMode.REQUIRED,
        tolerance=_ABS_RELAX,
        grad_patterns=_GRAD_PATTERNS,
        n_inputs=2,
        case_filter=_valid_broadcast_case,
        notes="CANCELLATION pattern：b = a + δ，验证相近值相减的有效位丢失。",
    )


@register
def mul():
    return OpSpec(
        name="mul",
        category="binaryeltwise",
        dtypes=[BF16, FP16, FP32],
        shapes=_BIN_SHAPES,
        patterns=[
            NORMAL, ZEROS, UNIFORM1, UNIFORM10, UNIFORM1000,
            EXTREME_VALUES(0.02),      # 0×inf / 极小×极大
            MASKFILL(0.0, 0.2),        # mask 调制 pattern
        ],
        params=_BCAST_PARAMS,
        reffn=lambda a, b: torch.mul(a, b),
        ppufn=lambda a, b: torch.mul(a, b),
        has_backward=BackwardMode.REQUIRED,
        tolerance=ToleranceOverride(
            max_abs_err=2.0e4,
            reason="mul 输出量级可达 ~1e6 (uniform1000²)，BF16 1-ULP 绝对误差随之增大"
            "(~8192)，绝对阈值放宽至 2e4；相对误差为主判据。0×inf=NaN 由硬性判据校验。",
        ),
        grad_patterns=_GRAD_PATTERNS,
        n_inputs=2,
        case_filter=_valid_broadcast_case,
        notes="覆盖 0×inf(NaN)、极小×极大、mask 填 0 调制。",
    )
