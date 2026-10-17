"""Tolerance table and dtype helpers (§4.3 of the spec).

Two responsibilities live here:

* a canonical mapping between the CLI/config dtype names (``bf16``, ``fp32``,
  ``fp16``, ``fp64``) and :class:`torch.dtype` objects, and
* the pass/fail threshold table plus the per-``OpSpec`` override logic.

The default thresholds reproduce the table in §4.3.  A "reduction" operator
(anything that accumulates across elements, e.g. Softmax) gets a relaxed FP32
absolute threshold (``<= 1e-4``) exactly as the spec calls out.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import torch

# --------------------------------------------------------------------------
# dtype registry
# --------------------------------------------------------------------------
BF16 = torch.bfloat16
FP16 = torch.float16
FP32 = torch.float32
FP64 = torch.float64

DTYPE_MAP: Dict[str, torch.dtype] = {
    "bf16": BF16,
    "bfloat16": BF16,
    "fp16": FP16,
    "float16": FP16,
    "half": FP16,
    "fp32": FP32,
    "float32": FP32,
    "fp64": FP64,
    "float64": FP64,
    "double": FP64,
}

_DTYPE_TO_NAME = {
    BF16: "bf16",
    FP16: "fp16",
    FP32: "fp32",
    FP64: "fp64",
}


def dtype_name(dtype: torch.dtype) -> str:
    """Short, stable, filesystem/CLI friendly name for a torch dtype."""
    return _DTYPE_TO_NAME.get(dtype, str(dtype).replace("torch.", ""))


def parse_dtype(token: str) -> torch.dtype:
    """Resolve a single dtype token (``bf16``/``fp32``/``fp16``/...)."""
    key = token.strip().lower()
    if key not in DTYPE_MAP:
        raise ValueError(
            f"unknown dtype {token!r}; expected one of {sorted(DTYPE_MAP)}"
        )
    return DTYPE_MAP[key]


def parse_dtypes(token) -> list:
    """Resolve a comma-separated dtype list or an already-parsed iterable."""
    if token is None:
        return []
    if isinstance(token, str):
        return [parse_dtype(t) for t in token.split(",") if t.strip()]
    out = []
    for t in token:
        out.append(t if isinstance(t, torch.dtype) else parse_dtype(str(t)))
    return out


# --------------------------------------------------------------------------
# tolerance table
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Tolerance:
    """Concrete pass/fail thresholds for a single (dtype, direction) cell."""

    max_abs_err: float
    max_rel_err: float
    min_cosine_sim: float
    max_mismatch_rate: float

    def as_dict(self) -> dict:
        return {
            "max_abs_err": self.max_abs_err,
            "max_rel_err": self.max_rel_err,
            "min_cosine_sim": self.min_cosine_sim,
            "max_mismatch_rate": self.max_mismatch_rate,
        }


# Default table (§4.3).  ``min_cosine_sim`` is stored as the concrete lower
# bound (``1 - slack``) so comparison code stays a simple ``>=``.
DEFAULT_TOLERANCE: Dict[torch.dtype, Tolerance] = {
    FP32: Tolerance(
        max_abs_err=1e-5,
        max_rel_err=1e-5,
        min_cosine_sim=1.0 - 1e-6,
        max_mismatch_rate=0.0001,  # 0.01 %
    ),
    BF16: Tolerance(
        max_abs_err=8e-3,  # ~1 ULP @ 1.0
        max_rel_err=1.6e-2,
        min_cosine_sim=1.0 - 1e-3,
        max_mismatch_rate=0.01,  # 1 %
    ),
    FP16: Tolerance(
        max_abs_err=1e-3,
        max_rel_err=2e-3,
        min_cosine_sim=1.0 - 1e-4,
        max_mismatch_rate=0.001,  # 0.1 %
    ),
    # FP64 is not a target dtype in the matrix but is handy for self-checks.
    FP64: Tolerance(
        max_abs_err=1e-12,
        max_rel_err=1e-12,
        min_cosine_sim=1.0 - 1e-12,
        max_mismatch_rate=0.0,
    ),
}

# Reduction ops accumulate in FP32, so the spec relaxes the abs threshold.
REDUCTION_ABS_ERR: Dict[torch.dtype, float] = {
    FP32: 1e-4,
}


@dataclass(frozen=True)
class ToleranceOverride:
    """Per-``OpSpec`` relaxation/tightening of the default table.

    Any field left as ``None`` keeps the default.  ``reduction=True`` opts into
    the relaxed reduction absolute threshold for dtypes that define one.  A
    non-empty ``reason`` is mandatory whenever a threshold is loosened; the
    report surfaces it next to the operator so relaxations stay auditable.
    """

    max_abs_err: Optional[float] = None
    max_rel_err: Optional[float] = None
    min_cosine_sim: Optional[float] = None
    max_mismatch_rate: Optional[float] = None
    reduction: bool = False
    reason: str = ""

    @property
    def loosens(self) -> bool:
        return any(
            v is not None
            for v in (
                self.max_abs_err,
                self.max_rel_err,
                self.min_cosine_sim,
                self.max_mismatch_rate,
            )
        ) or self.reduction


def resolve_tolerance(
    dtype: torch.dtype, override: Optional[ToleranceOverride] = None
) -> Tolerance:
    """Merge the default table with an optional ``OpSpec`` override."""
    base = DEFAULT_TOLERANCE.get(dtype)
    if base is None:
        raise ValueError(f"no default tolerance for dtype {dtype}")
    if override is None:
        return base

    merged = base
    if override.reduction and dtype in REDUCTION_ABS_ERR:
        merged = replace(
            merged,
            max_abs_err=max(merged.max_abs_err, REDUCTION_ABS_ERR[dtype]),
        )
    if override.max_abs_err is not None:
        merged = replace(merged, max_abs_err=override.max_abs_err)
    if override.max_rel_err is not None:
        merged = replace(merged, max_rel_err=override.max_rel_err)
    if override.min_cosine_sim is not None:
        merged = replace(merged, min_cosine_sim=override.min_cosine_sim)
    if override.max_mismatch_rate is not None:
        merged = replace(merged, max_mismatch_rate=override.max_mismatch_rate)
    return merged
