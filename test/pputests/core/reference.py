"""Golden (reference) computation (§4.1 of the spec).

Primary golden
    PyTorch **CPU float64**, result cast back to the target dtype.  Computing
    in fp64 avoids the reference itself being the precision bottleneck.

Secondary golden (cross-check)
    PyTorch CPU **native target dtype** (fp32/bf16).  Comparing the PPU output
    against *both* lets the report distinguish "PPU implementation problem"
    (fails vs fp64 but fp32-CPU matches fp64) from "inherent low-precision
    error" (fp32-CPU also drifts from fp64 by a similar amount).

The golden always consumes bit-identical inputs to the PPU run: the device
tensors are upcast to fp64, which is lossless for bf16/fp16/fp32.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch


@dataclass
class GoldenResult:
    out_fp64: torch.Tensor                 # raw fp64 forward output
    out_target: torch.Tensor               # fp64 -> cast to target dtype (primary)
    out_secondary: Optional[torch.Tensor]  # cpu native target-dtype forward
    inputs_fp64: List[torch.Tensor] = field(default_factory=list)


def to_fp64_inputs(
    inputs: Sequence[torch.Tensor], requires_grad: Sequence[bool]
) -> List[torch.Tensor]:
    """Clone device inputs onto CPU fp64, tagging grad-requiring leaves."""
    out = []
    for t, rg in zip(inputs, requires_grad):
        leaf = t.detach().to(device="cpu", dtype=torch.float64)
        if rg:
            leaf = leaf.requires_grad_(True)
        out.append(leaf)
    return out


def golden_forward(
    reffn,
    inputs_fp64: Sequence[torch.Tensor],
    call_params: dict,
    target_dtype: torch.dtype,
    *,
    secondary_inputs: Optional[Sequence[torch.Tensor]] = None,
    secondary_dtype: Optional[torch.dtype] = None,
) -> GoldenResult:
    """Run the golden forward in fp64 (+ optional same-dtype cross-check)."""
    out_fp64 = reffn(*inputs_fp64, **call_params)

    if isinstance(out_fp64, (tuple, list)):
        out_fp64 = out_fp64[0]
    out_target = out_fp64.detach().to(target_dtype)

    out_secondary = None
    if secondary_inputs is not None and secondary_dtype is not None:
        sec_in = [
            t.detach().to(device="cpu", dtype=secondary_dtype)
            for t in secondary_inputs
        ]
        sec = reffn(*sec_in, **call_params)
        if isinstance(sec, (tuple, list)):
            sec = sec[0]
        out_secondary = sec.detach()

    return GoldenResult(
        out_fp64=out_fp64,
        out_target=out_target,
        out_secondary=out_secondary,
        inputs_fp64=list(inputs_fp64),
    )


def golden_backward(
    golden: GoldenResult,
    grad_out_target_dtype: torch.Tensor,
) -> List[torch.Tensor]:
    """Backprop the fp64 golden graph; returns fp64 grads for the leaf inputs.

    ``grad_out_target_dtype`` is the same upstream gradient used on the PPU
    side (target dtype); it is upcast to fp64 to drive the golden backward.
    """
    leaves = [t for t in golden.inputs_fp64 if t.requires_grad]
    if not leaves:
        return []
    grad_fp64 = grad_out_target_dtype.detach().to(
        device="cpu", dtype=torch.float64
    )
    grads = torch.autograd.grad(
        golden.out_fp64, leaves, grad_outputs=grad_fp64, allow_unused=True
    )
    return [
        (g if g is not None else torch.zeros_like(l))
        for g, l in zip(grads, leaves)
    ]
