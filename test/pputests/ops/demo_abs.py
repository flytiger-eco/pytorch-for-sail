"""Demo operator (DoD #5): prove extensibility with <= 20 lines, 0 framework edits.

Drop this file into ``ops/`` and it is auto-registered -- no other file changes.
"""

from __future__ import annotations

import torch

from core.inputgen import EXTREME_VALUES, NEARBOUNDARY, NORMAL, ZEROS, preset_shapes
from core.opspec import BackwardMode, OpSpec, register
from core.tolerance import BF16, FP32


@register
def demo_abs():
    return OpSpec(
        name="abs", category="demo",
        dtypes=[BF16, FP32],
        shapes=preset_shapes("activation"),
        patterns=[NORMAL, ZEROS, NEARBOUNDARY(values=[0.0], ulp_span=2),
                  EXTREME_VALUES(0.01)],
        params=[{}],
        reffn=lambda x: torch.abs(x),
        ppufn=lambda x: torch.abs(x),
        has_backward=BackwardMode.REQUIRED,
        notes="demo：x=0 处梯度取 torch 语义 (0)。",
    )
