"""Comparison metrics and pass/fail judgement (§4.2 / §4.3 of the spec).

All arithmetic is performed in **fp64** so the metric computation itself never
becomes the limiting precision.  Two families of checks exist:

* *soft* metrics compared against a :class:`~core.tolerance.Tolerance`
  (max abs/rel err, cosine similarity, mismatch rate, plus trend-only metrics),
* *hard* rules that always fail a case regardless of thresholds -- NaN/Inf
  position & sign mismatch, and shape/dtype/stride divergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .tolerance import Tolerance, dtype_name

_EPS = 1e-12


@dataclass
class Comparison:
    """Result of comparing one PPU tensor against its golden reference."""

    metrics: Dict[str, float] = field(default_factory=dict)
    hard_ok: bool = True
    hard_reasons: List[str] = field(default_factory=list)
    soft_reasons: List[str] = field(default_factory=list)
    passed: bool = True
    # Top-K offenders (flat index, out value, ref value, abs err) for reports.
    top_offenders: List[tuple] = field(default_factory=list)

    @property
    def reasons(self) -> List[str]:
        return self.hard_reasons + self.soft_reasons


def _to_fp64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float64)


def _special_consistency(out: torch.Tensor, ref: torch.Tensor) -> List[str]:
    """Hard rule: NaN/Inf positions AND inf signs must match exactly."""
    reasons: List[str] = []
    o64, r64 = _to_fp64(out), _to_fp64(ref)

    o_nan, r_nan = torch.isnan(o64), torch.isnan(r64)
    if not torch.equal(o_nan, r_nan):
        n = int((o_nan ^ r_nan).sum().item())
        reasons.append(f"NaN position mismatch ({n} elements)")

    o_inf, r_inf = torch.isinf(o64), torch.isinf(r64)
    if not torch.equal(o_inf, r_inf):
        n = int((o_inf ^ r_inf).sum().item())
        reasons.append(f"Inf position mismatch ({n} elements)")
    else:
        # same inf positions -> compare signs
        o_pos, r_pos = torch.isposinf(o64), torch.isposinf(r64)
        if not torch.equal(o_pos, r_pos):
            reasons.append("Inf sign mismatch")
    return reasons


def _ulp_err_max(out: torch.Tensor, ref: torch.Tensor) -> Optional[float]:
    """Optional strict metric: max ULP distance over finite, same-sign pairs.

    Only defined for fp32/fp16 (bit-level layout); returns ``None`` otherwise
    (e.g. bf16) or when numpy is unavailable.
    """
    if out.dtype not in (torch.float32, torch.float16):
        return None
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return None
    o = out.detach().to("cpu").contiguous().numpy()
    r = ref.detach().to("cpu").contiguous().numpy()
    if o.shape != r.shape:
        return None
    finite = np.isfinite(o) & np.isfinite(r)
    if not finite.any():
        return 0.0
    ov = o[finite].astype(np.float32).view(np.int32)
    rv = r[finite].astype(np.float32).view(np.int32)
    # map sign-magnitude ints to a monotonic scale so |a-b| == ULP distance
    def _key(x):
        return np.where(x < 0, np.int32(-2147483648) - x, x)
    dist = np.abs(_key(ov).astype(np.int64) - _key(rv).astype(np.int64))
    return float(dist.max())


def compare(
    out: torch.Tensor,
    ref: torch.Tensor,
    tol: Tolerance,
    *,
    dtype: torch.dtype,
    check_shape: bool = True,
    check_dtype: bool = True,
    check_stride: bool = False,
    empty_is_pass: bool = True,
    topk: int = 8,
) -> Comparison:
    """Compare ``out`` (PPU) against ``ref`` (golden) under ``tol``."""
    cmp = Comparison()

    # ---- hard: shape / dtype / stride ------------------------------------
    if check_shape and tuple(out.shape) != tuple(ref.shape):
        cmp.hard_ok = False
        cmp.hard_reasons.append(
            f"shape mismatch: ppu {tuple(out.shape)} vs ref {tuple(ref.shape)}"
        )
        cmp.passed = False
        return cmp
    if check_dtype and out.dtype != dtype:
        cmp.hard_ok = False
        cmp.hard_reasons.append(
            f"dtype mismatch: ppu {dtype_name(out.dtype)} expected {dtype_name(dtype)}"
        )
        cmp.passed = False
        return cmp
    if check_stride and out.stride() != ref.stride():
        cmp.hard_ok = False
        cmp.hard_reasons.append(
            f"stride mismatch: ppu {out.stride()} vs ref {ref.stride()}"
        )
        cmp.passed = False
        return cmp

    # ---- empty tensors: not crashing is the pass criterion ---------------
    if out.numel() == 0:
        cmp.metrics = {
            "numel": 0.0,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "mean_abs_err": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "cosine_sim": 1.0,
            "mismatch_rate": 0.0,
        }
        cmp.passed = bool(empty_is_pass)
        if not empty_is_pass:
            cmp.soft_reasons.append("empty tensor but empty_is_pass=False")
        return cmp

    # ---- hard: NaN / Inf consistency -------------------------------------
    special = _special_consistency(out, ref)
    if special:
        cmp.hard_ok = False
        cmp.hard_reasons.extend(special)

    # ---- soft metrics (finite elements only) -----------------------------
    o64, r64 = _to_fp64(out), _to_fp64(ref)
    finite = torch.isfinite(o64) & torch.isfinite(r64)
    of = o64[finite]
    rf = r64[finite]
    n = of.numel()

    if n == 0:
        # every element is special -> only the hard NaN/Inf rule applies
        cmp.metrics = {
            "numel": float(o64.numel()),
            "finite": 0.0,
            "max_abs_err": float("nan"),
            "max_rel_err": float("nan"),
            "mean_abs_err": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "cosine_sim": float("nan"),
            "mismatch_rate": 1.0 if not cmp.hard_ok else 0.0,
        }
        cmp.passed = cmp.hard_ok
        return cmp

    diff = (of - rf).abs()
    denom = rf.abs().clamp_min(_EPS)
    rel = diff / denom

    max_abs = float(diff.max().item())
    max_rel = float(rel.max().item())
    mean_abs = float(diff.mean().item())
    mse = float((diff.pow(2)).mean().item())
    rmse = math.sqrt(mse)

    # cosine similarity over the flattened finite vectors
    onorm = float(of.norm().item())
    rnorm = float(rf.norm().item())
    if onorm == 0.0 and rnorm == 0.0:
        cos = 1.0
    elif onorm == 0.0 or rnorm == 0.0:
        cos = 0.0
    else:
        cos = float(torch.dot(of, rf).item() / (onorm * rnorm))

    # mismatch = element violates BOTH the abs and rel budget (isclose style)
    bad = (diff > tol.max_abs_err) & (rel > tol.max_rel_err)
    mismatch_rate = float(bad.to(torch.float64).mean().item())

    cmp.metrics = {
        "numel": float(o64.numel()),
        "finite": float(n),
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "mean_abs_err": mean_abs,
        "mse": mse,
        "rmse": rmse,
        "cosine_sim": cos,
        "mismatch_rate": mismatch_rate,
    }
    ulp = _ulp_err_max(out, ref)
    if ulp is not None:
        cmp.metrics["ulp_err_max"] = ulp

    # ---- soft threshold judgement ----------------------------------------
    if max_abs > tol.max_abs_err:
        cmp.soft_reasons.append(
            f"max_abs_err {max_abs:.3e} > {tol.max_abs_err:.3e}"
        )
    if max_rel > tol.max_rel_err:
        cmp.soft_reasons.append(
            f"max_rel_err {max_rel:.3e} > {tol.max_rel_err:.3e}"
        )
    if cos < tol.min_cosine_sim:
        cmp.soft_reasons.append(
            f"cosine_sim {cos:.9f} < {tol.min_cosine_sim:.9f}"
        )
    if mismatch_rate > tol.max_mismatch_rate:
        cmp.soft_reasons.append(
            f"mismatch_rate {mismatch_rate:.4%} > {tol.max_mismatch_rate:.4%}"
        )

    # ---- Top-K offenders (helps triage "outlier vs global drift") --------
    k = min(topk, n)
    if k > 0:
        vals, idx = torch.topk(diff.reshape(-1), k)
        flat_o = of.reshape(-1)
        flat_r = rf.reshape(-1)
        for j, pos in enumerate(idx.tolist()):
            cmp.top_offenders.append(
                (
                    int(pos),
                    float(flat_o[pos].item()),
                    float(flat_r[pos].item()),
                    float(vals[j].item()),
                )
            )

    cmp.passed = cmp.hard_ok and not cmp.soft_reasons
    return cmp
