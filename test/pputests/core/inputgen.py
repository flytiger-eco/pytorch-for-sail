"""Shape / Pattern input generation (§5 of the spec).

Inputs are produced by a three-layer combination that forms the case matrix:

* **Shape layer** (:class:`ShapeCase`) -- regular, business-real, large,
  non-contiguous, extreme-dimension and empty tensors.
* **Numeric-distribution layer** (:class:`Pattern`) -- normal, uniform,
  constants, corner sweeps, extreme/special values, near-boundary ULP spans,
  catastrophic-cancellation pairs, mask fills and reduction-order-sensitive
  orderings.
* **Parameter layer** -- handled by the ``OpSpec`` ``params`` list; a few
  reserved keys (``_shape_b``, ``_view_b``, ``_pattern_b``) drive the *second*
  operand of binary operators and are stripped before the op is called.

Tensors are materialised directly in the **target dtype** so that the golden
reference sees bit-identical inputs (upcast to fp64) -- this keeps the
comparison honest for low-precision dtypes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------
# Shape layer (§5.1)
# --------------------------------------------------------------------------
# ``view`` describes a memory-layout transformation applied after generation:
#   None        -> contiguous
#   transpose12 -> t.transpose(1, 2)   (needs >= 3 dims)
#   slice2      -> t[..., ::2]         (strided slice on last dim)
#   expand      -> expand a size-1 dim (broadcast view, stride 0)
VALID_VIEWS = (None, "transpose12", "slice2", "expand")


@dataclass(frozen=True)
class ShapeCase:
    """One entry of the shape layer."""

    name: str
    shape: Tuple[int, ...]
    category: str = "regular"
    view: Optional[str] = None

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def _sc(*shape: int) -> Tuple[int, ...]:
    return tuple(shape)


# The six mandatory shape categories (§5.1).  ``preset_shapes`` composes a
# subset per operator family; every category is exercised by at least one op.
REGULAR_SHAPES: List[ShapeCase] = [
    ShapeCase("scalar_1d", _sc(1), "regular"),
    ShapeCase("vec16", _sc(16), "regular"),
    ShapeCase("mat4x4", _sc(4, 4), "regular"),
    ShapeCase("box_8_16_32", _sc(8, 16, 32), "regular"),
]

BUSINESS_SHAPES: List[ShapeCase] = [
    ShapeCase("vae_2_4_64_64", _sc(2, 4, 64, 64), "business"),
    ShapeCase("attn_2_8_512_64", _sc(2, 8, 512, 64), "business"),
    ShapeCase("vlm_4_1024_4096", _sc(4, 1024, 4096), "business"),
]

LARGE_SHAPES: List[ShapeCase] = [
    # 2 * 4096 * 2048 == 2**24 elements -> triggers tiling / multi-core paths.
    ShapeCase("large_2pow24", _sc(2, 4096, 2048), "large"),
]

NONCONTIG_SHAPES: List[ShapeCase] = [
    ShapeCase("transpose12_4_8_16", _sc(4, 8, 16), "noncontig", "transpose12"),
    ShapeCase("slice2_32_64", _sc(32, 64), "noncontig", "slice2"),
    ShapeCase("expand_4_1_16", _sc(4, 1, 16), "noncontig", "expand"),
]

EXTREME_DIM_SHAPES: List[ShapeCase] = [
    ShapeCase("dim1_1_8_1", _sc(1, 8, 1), "extreme_dim"),
    ShapeCase("prime_4093", _sc(4093), "extreme_dim"),
    ShapeCase("prime_2d_13_4093", _sc(13, 4093), "extreme_dim"),
    ShapeCase("wide_1_1pow20", _sc(1, 1 << 20), "extreme_dim"),
]

EMPTY_SHAPES: List[ShapeCase] = [
    ShapeCase("empty_0", _sc(0), "empty"),
    ShapeCase("empty_2d_0_4", _sc(0, 4), "empty"),
]

ALL_SHAPE_CATEGORIES = {
    "regular": REGULAR_SHAPES,
    "business": BUSINESS_SHAPES,
    "large": LARGE_SHAPES,
    "noncontig": NONCONTIG_SHAPES,
    "extreme_dim": EXTREME_DIM_SHAPES,
    "empty": EMPTY_SHAPES,
}


def preset_shapes(kind: str = "unary", *, large: bool = True) -> List[ShapeCase]:
    """Return a shape preset for an operator family.

    ``kind`` is one of ``unary`` / ``activation`` / ``reduction`` / ``binary``.
    The presets all cover the six mandatory categories; ``binary`` adds a few
    4-D shapes so broadcast parameter combos have something to bite on.
    """
    base: List[ShapeCase] = []
    base += REGULAR_SHAPES
    base += BUSINESS_SHAPES[:2]
    base += NONCONTIG_SHAPES
    base += EXTREME_DIM_SHAPES
    base += EMPTY_SHAPES
    if large:
        base += LARGE_SHAPES

    if kind == "reduction":
        # reductions want a clearly-reducible trailing dim + a middle dim.
        base += [
            ShapeCase("reduce_4_8_16", _sc(4, 8, 16), "regular"),
            ShapeCase("reduce_2_8_512_64", _sc(2, 8, 512, 64), "business"),
        ]
    if kind == "binary":
        base += [
            ShapeCase("nchw_2_4_8_8", _sc(2, 4, 8, 8), "business"),
            ShapeCase("nc_8_16", _sc(8, 16), "regular"),
        ]
    # de-dup by name, keep order
    seen = set()
    out = []
    for sc in base:
        if sc.name in seen:
            continue
        seen.add(sc.name)
        out.append(sc)
    return out


# --------------------------------------------------------------------------
# Pattern layer (§5.2)
# --------------------------------------------------------------------------
class PatternKind(Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    ZEROS = "zeros"
    ONES = "ones"
    FULL = "full"
    Linspace_CORNER = "linspace_corner"
    EXTREME = "extreme"
    NEARBOUNDARY = "nearboundary"
    CANCELLATION = "cancellation"
    MASKFILL = "maskfill"
    ALTERNATING = "alternating"
    SORTED_ASC = "sorted_asc"
    SORTED_DESC = "sorted_desc"


@dataclass(frozen=True)
class Pattern:
    """A numeric-distribution recipe (optionally parameterised)."""

    kind: PatternKind
    # generic amplitude used by UNIFORM / FULL / magnitude of structured fills
    a: float = 1.0
    # near-boundary corner values and their ULP span
    values: Tuple[float, ...] = ()
    ulp_span: int = 2
    # fraction of special values injected by EXTREME / MASKFILL
    ratio: float = 0.0
    fill: float = float("-inf")
    # cancellation delta
    delta: float = 1e-6
    # linspace sweep range
    lo: float = -1.0
    hi: float = 1.0

    @property
    def label(self) -> str:
        k = self.kind.value
        if self.kind is PatternKind.UNIFORM:
            return f"uniform{self.a:g}"
        if self.kind is PatternKind.FULL:
            return f"full{self.a:g}"
        if self.kind is PatternKind.Linspace_CORNER:
            return f"linspace[{self.lo:g},{self.hi:g}]"
        if self.kind is PatternKind.NEARBOUNDARY:
            vals = "_".join(f"{v:g}" for v in self.values)
            return f"nearboundary[{vals}]"
        if self.kind is PatternKind.MASKFILL:
            return f"maskfill[{self.fill:g}x{self.ratio:g}]"
        if self.kind is PatternKind.EXTREME:
            return f"extreme[{self.ratio:g}]"
        if self.kind is PatternKind.CANCELLATION:
            return f"cancellation[{self.delta:g}]"
        return k


# Pre-instantiated constants used directly in OpSpec.patterns.
NORMAL = Pattern(PatternKind.NORMAL)
ZEROS = Pattern(PatternKind.ZEROS)
ONES = Pattern(PatternKind.ONES)
UNIFORM1 = Pattern(PatternKind.UNIFORM, a=1.0)
UNIFORM10 = Pattern(PatternKind.UNIFORM, a=10.0)
UNIFORM1000 = Pattern(PatternKind.UNIFORM, a=1000.0)
ALTERNATING = Pattern(PatternKind.ALTERNATING, a=1.0)
SORTED_ASC = Pattern(PatternKind.SORTED_ASC)
SORTED_DESC = Pattern(PatternKind.SORTED_DESC)
CANCELLATION = Pattern(PatternKind.CANCELLATION, delta=1e-6)
EXTREME = Pattern(PatternKind.EXTREME, ratio=0.0)  # special values off by default


# Factory helpers (the ``NEARBOUNDARY(values=[...], ulp_span=2)`` style).
def UNIFORM(a: float) -> Pattern:  # noqa: N802 - factory reads as a constant
    return Pattern(PatternKind.UNIFORM, a=float(a))


def FULL(c: float) -> Pattern:  # noqa: N802
    return Pattern(PatternKind.FULL, a=float(c))


def Linspace(lo: float, hi: float) -> Pattern:  # noqa: N802
    return Pattern(PatternKind.Linspace_CORNER, lo=float(lo), hi=float(hi))


def NEARBOUNDARY(values: Sequence[float], ulp_span: int = 2) -> Pattern:  # noqa: N802
    return Pattern(
        PatternKind.NEARBOUNDARY,
        values=tuple(float(v) for v in values),
        ulp_span=int(ulp_span),
    )


def EXTREME_VALUES(ratio: float = 0.01) -> Pattern:  # noqa: N802
    return Pattern(PatternKind.EXTREME, ratio=float(ratio))


def MASKFILL(fill: float = float("-inf"), ratio: float = 0.1) -> Pattern:  # noqa: N802
    return Pattern(PatternKind.MASKFILL, fill=float(fill), ratio=float(ratio))


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def _finfo(dtype: torch.dtype):
    return torch.finfo(dtype)


def _nextafter_value(value: float, dtype: torch.dtype, positive: bool) -> float:
    """Return the adjacent ``dtype`` value towards +/- infinity."""
    t = torch.tensor(float(value), dtype=dtype)
    direction = float("inf") if positive else float("-inf")
    try:
        return float(torch.nextafter(t, torch.full_like(t, direction)).item())
    except (RuntimeError, TypeError):
        # IEEE-style fallback for CPU builds missing low-precision nextafter.
        x = float(t.item())
        if math.isnan(x) or (math.isinf(x) and (x > 0.0) == positive):
            return x
        fi = _finfo(dtype)
        min_step = float(fi.tiny * fi.eps)
        if x == 0.0:
            return min_step if positive else -min_step

        exponent = math.floor(math.log2(abs(x)))
        step = max(float(math.ldexp(fi.eps, exponent)), min_step)
        # At an exact power-of-two boundary, the interval towards zero belongs
        # to the preceding binade and is half the outward interval.
        exact_power = abs(x) == math.ldexp(1.0, exponent)
        moving_toward_zero = (x > 0.0 and not positive) or (x < 0.0 and positive)
        if exact_power and moving_toward_zero:
            step *= 0.5
        return x + step if positive else x - step


def _ulp_at(value: float, dtype: torch.dtype) -> float:
    """Return one representable ``dtype`` step towards positive infinity."""
    center = float(torch.tensor(float(value), dtype=dtype).item())
    return abs(_nextafter_value(center, dtype, positive=True) - center)


def effective_shape(
    shape: Tuple[int, ...], view: Optional[str] = None
) -> Tuple[int, ...]:
    """Compute the logical shape produced by ``_apply_view`` without a tensor."""
    target = list(shape)
    if view is None:
        return tuple(target)
    if view == "transpose12":
        if len(target) >= 3:
            target[1], target[2] = target[2], target[1]
        return tuple(target)
    if view == "slice2":
        if target and target[-1] >= 2:
            target[-1] = (target[-1] + 1) // 2
        return tuple(target)
    if view == "expand":
        for dim, size in enumerate(target):
            if size == 1:
                target[dim] = 4
                break
        return tuple(target)
    raise ValueError(f"unknown view {view!r}")


def _apply_view(t: torch.Tensor, view: Optional[str]) -> torch.Tensor:
    if view is None:
        return t
    if view == "transpose12":
        if t.dim() < 3:
            return t
        return t.transpose(1, 2)
    if view == "slice2":
        if t.dim() == 0 or t.shape[-1] < 2:
            return t
        return t[..., ::2]
    if view == "expand":
        # find a size-1 dim (or make the leading dim broadcastable) and expand
        for d in range(t.dim()):
            if t.shape[d] == 1:
                target = list(t.shape)
                target[d] = 4
                return t.expand(*target)
        return t
    raise ValueError(f"unknown view {view!r}")


def _base_random(
    pattern: Pattern,
    numel: int,
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
) -> torch.Tensor:
    """The 'core' values of a pattern before any special-value injection."""
    if numel == 0:
        return torch.empty(0, dtype=dtype, device=device)

    if pattern.kind is PatternKind.NORMAL:
        t = torch.randn(numel, generator=gen, dtype=torch.float32)
    elif pattern.kind is PatternKind.UNIFORM:
        a = pattern.a
        t = torch.rand(numel, generator=gen, dtype=torch.float32) * (2 * a) - a
    elif pattern.kind in (PatternKind.ZEROS,):
        t = torch.zeros(numel, dtype=torch.float32)
    elif pattern.kind in (PatternKind.ONES,):
        t = torch.ones(numel, dtype=torch.float32)
    elif pattern.kind is PatternKind.FULL:
        t = torch.full((numel,), pattern.a, dtype=torch.float32)
    elif pattern.kind is PatternKind.Linspace_CORNER:
        t = torch.linspace(pattern.lo, pattern.hi, numel, dtype=torch.float32)
    elif pattern.kind is PatternKind.NEARBOUNDARY:
        t = _nearboundary_values(pattern, numel, dtype, gen)
    elif pattern.kind is PatternKind.ALTERNATING:
        idx = torch.arange(numel, dtype=torch.float32)
        sign = torch.where(idx % 2 == 0, 1.0, -1.0)
        mag = torch.rand(numel, generator=gen, dtype=torch.float32) * pattern.a
        t = sign * mag
    elif pattern.kind in (PatternKind.SORTED_ASC, PatternKind.SORTED_DESC):
        t = torch.linspace(-pattern.a if pattern.a else -1.0,
                           pattern.a if pattern.a else 1.0,
                           numel, dtype=torch.float32)
        if pattern.kind is PatternKind.SORTED_DESC:
            t = torch.flip(t, [0])
    elif pattern.kind in (PatternKind.CANCELLATION, PatternKind.MASKFILL,
                          PatternKind.EXTREME):
        # these start from a random/normal base then get modified below
        t = torch.randn(numel, generator=gen, dtype=torch.float32)
    else:  # pragma: no cover - guarded by enum
        raise ValueError(f"unsupported pattern {pattern.kind}")

    return t.to(device=device, dtype=dtype)


def _nearboundary_values(
    pattern: Pattern, numel: int, dtype: torch.dtype, gen: torch.Generator
) -> torch.Tensor:
    """Densely sample ``values`` +/- k ULP so corner gradients are exercised."""
    values = pattern.values or (0.0,)
    span = max(1, pattern.ulp_span)
    pts: List[float] = []
    for value in values:
        center = float(torch.tensor(value, dtype=dtype).item())
        below: List[float] = []
        current = center
        for _ in range(span):
            current = _nextafter_value(current, dtype, positive=False)
            below.append(current)
        above: List[float] = []
        current = center
        for _ in range(span):
            current = _nextafter_value(current, dtype, positive=True)
            above.append(current)
        pts.extend(reversed(below))
        pts.append(center)
        pts.extend(above)
        pts.append(center)  # weight the exact corner as well as its neighbours
    # tile / trim to numel
    reps = (numel + len(pts) - 1) // len(pts)
    tiled = (pts * reps)[:numel]
    return torch.tensor(tiled, dtype=torch.float32)


def _inject_special(
    t: torch.Tensor, pattern: Pattern, dtype: torch.dtype, gen: torch.Generator
) -> torch.Tensor:
    """Overwrite a ``ratio`` fraction of elements with special/extreme values."""
    if t.numel() == 0 or pattern.ratio <= 0.0:
        return t
    fi = _finfo(dtype)
    specials = [
        fi.max,
        -fi.max,
        float("inf"),
        float("-inf"),
        float("nan"),
        fi.tiny,
        -fi.tiny,
        fi.smallest_subnormal if hasattr(fi, "smallest_subnormal") else 0.0,
    ]
    flat = t.reshape(-1).clone()
    n = flat.numel()
    k = max(1, int(round(n * pattern.ratio)))
    idx = torch.randperm(n, generator=gen)[:k].to(flat.device)
    picks = torch.tensor(
        [specials[i % len(specials)] for i in range(k)], dtype=torch.float32
    )
    flat[idx] = picks.to(device=flat.device, dtype=flat.dtype)
    return flat.reshape(t.shape)


def _apply_maskfill(
    t: torch.Tensor, pattern: Pattern, dtype: torch.dtype, gen: torch.Generator
) -> torch.Tensor:
    if t.numel() == 0 or pattern.ratio <= 0.0:
        return t
    flat = t.reshape(-1).clone()
    n = flat.numel()
    k = max(1, int(round(n * pattern.ratio)))
    idx = torch.randperm(n, generator=gen)[:k].to(flat.device)
    fill = torch.full((k,), pattern.fill, dtype=torch.float32)
    flat[idx] = fill.to(device=flat.device, dtype=flat.dtype)
    return flat.reshape(t.shape)


def generate_input(
    pattern: Pattern,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
    view: Optional[str] = None,
    index: int = 0,
    base: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Materialise one input tensor for ``pattern`` at ``shape``/``dtype``.

    ``index`` and ``base`` support paired patterns: for CANCELLATION the second
    operand (``index >= 1``) is ``base + delta`` so a catastrophic-cancellation
    subtraction is exercised.
    """
    numel = 1
    for d in shape:
        numel *= d

    # cancellation: second operand derived from the already-materialised first
    if pattern.kind is PatternKind.CANCELLATION and index >= 1 and base is not None:
        desired_shape = effective_shape(shape, view)
        if tuple(base.shape) != desired_shape:
            raise ValueError(
                "CANCELLATION requires operands with the same effective shape; "
                f"got first={tuple(base.shape)} second={desired_shape}"
            )
        b = base.detach().to(device=device, dtype=dtype)
        return (b.to(torch.float32) + pattern.delta).to(dtype)

    t = _base_random(pattern, numel, dtype, device, gen)
    t = t.reshape(shape)

    if pattern.kind is PatternKind.EXTREME:
        t = _inject_special(t, pattern, dtype, gen)
    elif pattern.kind is PatternKind.MASKFILL:
        t = _apply_maskfill(t, pattern, dtype, gen)

    return _apply_view(t, view)


def build_inputs(
    patterns: Sequence[Pattern],
    shapes: Sequence[Tuple[int, ...]],
    views: Sequence[Optional[str]],
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
) -> List[torch.Tensor]:
    """Build the full input tuple for one case (unary or binary)."""
    tensors: List[torch.Tensor] = []
    base: Optional[torch.Tensor] = None
    for i, shape in enumerate(shapes):
        pat = patterns[i] if i < len(patterns) else patterns[-1]
        view = views[i] if i < len(views) else None
        t = generate_input(pat, shape, dtype, device, gen, view, index=i, base=base)
        if base is None:
            base = t
        tensors.append(t)
    return tensors


# --------------------------------------------------------------------------
# grad_output patterns (§5.3): at least normal / extreme / zeros
# --------------------------------------------------------------------------
DEFAULT_GRAD_PATTERNS: List[Pattern] = [NORMAL, ZEROS, EXTREME_VALUES(0.01)]


def grad_output_for(
    out_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
    pattern: Pattern = NORMAL,
) -> torch.Tensor:
    """Construct the upstream gradient fed into ``backward``."""
    return generate_input(pattern, tuple(out_shape), dtype, device, gen)
