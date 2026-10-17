"""``OpSpec`` dataclass + the operator registry (§6.2 of the spec).

Registering a new operator is the *only* change point required to extend the
framework: decorate a factory that returns an :class:`OpSpec`.  Nothing in
``core/`` or ``tests/`` needs to be touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .inputgen import (  # noqa: F401 (Pattern/ShapeCase/preset_shapes re-exported)
    NORMAL,
    Pattern,
    ShapeCase,
    preset_shapes,
)
from .tolerance import ToleranceOverride


class BackwardMode(Enum):
    """Whether/how the gradient path is validated for an operator."""

    REQUIRED = "required"  # 必测反向
    OPTIONAL = "optional"  # 参考侧可导，按需测
    NONE = "none"          # 无反向


# Reserved parameter keys consumed by the input builder, not passed to the op.
RESERVED_PARAM_KEYS = ("_shape_b", "_view_b", "_pattern_b", "_shapes", "_views")


@dataclass
class OpSpec:
    """Declarative description of one operator under test."""

    name: str                                   # e.g. "relu" / "softmax" / "sum"
    category: str                               # e.g. "activation"
    dtypes: List[torch.dtype]                   # [BF16, FP32]
    shapes: List[ShapeCase]                     # reuse inputgen presets or custom
    patterns: List[Pattern]                     # reuse inputgen presets
    params: List[dict]                          # op attribute combos, e.g. {"dim": -1}
    reffn: Callable                             # golden (runs on cpu fp64 inputs)
    ppufn: Callable                             # PPU-side call wrapper
    has_backward: BackwardMode = BackwardMode.NONE
    tolerance: Optional[ToleranceOverride] = None
    notes: str = ""                             # known deviations / relax reasons

    # how many tensor operands the op consumes (1 = unary, 2 = binary)
    n_inputs: int = 1
    # patterns used to build grad_output for the backward path (§5.3)
    grad_patterns: List[Pattern] = field(default_factory=lambda: [NORMAL])
    # validate output stride semantics in addition to shape/dtype
    check_stride: bool = False
    # empty tensors (numel == 0) only need to not crash -> no numeric compare
    empty_is_pass: bool = True
    # Optional predicate for excluding invalid shape/pattern/parameter products.
    case_filter: Optional[Callable[[ShapeCase, Pattern, dict], bool]] = None

    def call_params(self, params: dict) -> dict:
        """Strip reserved input-builder keys -> kwargs for reffn/ppufn."""
        return {k: v for k, v in params.items() if k not in RESERVED_PARAM_KEYS}

    def input_shapes(self, shape_case: ShapeCase, params: dict) -> List[tuple]:
        """Resolve the operand shapes for a case (handles binary broadcast)."""
        primary = tuple(shape_case.shape)
        if self.n_inputs <= 1:
            return [primary]
        if "_shapes" in params:
            return [tuple(s) for s in params["_shapes"]]
        shape_b = params.get("_shape_b", primary)
        return [primary, tuple(shape_b)]

    def input_views(self, shape_case: ShapeCase, params: dict) -> List[Optional[str]]:
        primary = shape_case.view
        if self.n_inputs <= 1:
            return [primary]
        if "_views" in params:
            return list(params["_views"])
        # For an implicit same-shape operand, preserve the same layout view.
        # Explicit broadcast shapes default to a contiguous second operand.
        default_b = primary if "_shape_b" not in params else None
        return [primary, params.get("_view_b", default_b)]

    def input_patterns(self, pattern: Pattern, params: dict) -> List[Pattern]:
        if self.n_inputs <= 1:
            return [pattern]
        pb = params.get("_pattern_b", pattern)
        return [pattern, pb]

    def supports_case(
        self, shape_case: ShapeCase, pattern: Pattern, params: dict
    ) -> bool:
        """Return whether a cartesian-product entry is meaningful for this op."""
        return self.case_filter is None or bool(
            self.case_filter(shape_case, pattern, params)
        )


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
OP_REGISTRY: Dict[str, OpSpec] = {}


def register(fn=None, *, replace: bool = False):
    """Register an :class:`OpSpec`.

    Usable bare (``@register``) on a zero-arg factory, or called
    (``@register()``).  The factory is invoked once at import time and its
    returned ``OpSpec`` is stored in :data:`OP_REGISTRY` keyed by ``name``.
    """

    def _do(factory):
        spec = factory()
        if not isinstance(spec, OpSpec):
            raise TypeError(
                f"@register factory {getattr(factory, '__name__', factory)!r} "
                "must return an OpSpec"
            )
        if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.name):
            raise ValueError(
                f"OpSpec name {spec.name!r} must be a generic lowercase "
                "operator name such as 'relu', 'softmax' or 'native_layer_norm'"
            )
        if spec.name in OP_REGISTRY and not replace:
            raise ValueError(f"duplicate OpSpec name {spec.name!r}")
        OP_REGISTRY[spec.name] = spec
        return spec

    if fn is None:
        return _do
    return _do(fn)


def get_spec(name: str) -> OpSpec:
    if name not in OP_REGISTRY:
        raise KeyError(
            f"unknown op {name!r}; registered: {sorted(OP_REGISTRY)}"
        )
    return OP_REGISTRY[name]


def resolve_spec_names(names: Sequence[str]) -> List[str]:
    """Resolve exact, case-insensitive operator names from user input.

    Requirement IDs and prefixes are intentionally not aliases. For example,
    users select ``softmax`` (not ``O25``); a future Sum spec is selected with
    ``sum`` without any framework change.
    """
    by_lower = {name.lower(): name for name in OP_REGISTRY}
    resolved: List[str] = []
    unknown: List[str] = []
    for token in names:
        key = token.strip().lower()
        if not key:
            continue
        if key not in by_lower:
            unknown.append(token)
            continue
        canonical = by_lower[key]
        if canonical not in resolved:
            resolved.append(canonical)
    if unknown:
        raise ValueError(
            f"unknown operator(s): {', '.join(unknown)}; "
            f"available operators: {', '.join(sorted(OP_REGISTRY))}"
        )
    return resolved


def all_specs() -> List[OpSpec]:
    return [OP_REGISTRY[k] for k in sorted(OP_REGISTRY)]


def specs_by_category(category: str) -> List[OpSpec]:
    return [s for s in all_specs() if s.category == category]
