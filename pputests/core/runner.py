"""Case enumeration, device resolution and execution engine (§6.3 of the spec).

Pipeline for one case::

    build inputs (target dtype, on PPU device)
      -> PPU forward            -> Golden forward (fp64 -> cast)
      -> metrics + tolerance    -> PASS/FAIL
      -> [if has_backward] build grad_output
             -> PPU backward    -> Golden backward
             -> metrics + tolerance -> PASS/FAIL

This module also owns the small YAML config loader shared by ``run.py`` and
``conftest.py`` so the two entry points stay in lock-step.
"""

from __future__ import annotations

import os
import random
import time
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from . import inputgen
from .inputgen import Pattern
from .metrics import Comparison, compare
from .opspec import OP_REGISTRY, BackwardMode, OpSpec, all_specs
from .reference import golden_backward, golden_forward, to_fp64_inputs
from .tolerance import (
    Tolerance,
    dtype_name,
    parse_dtypes,
    resolve_tolerance,
)

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "default.yaml",
)


@dataclass
class Config:
    dtypes: List[str] = field(default_factory=lambda: ["bf16", "fp32"])
    level: str = "smoke"                 # smoke | full
    seed: int = 0                        # int, or the string "random"
    device: str = "cuda"                 # PPU is exposed as a cuda-class device
    directions: List[str] = field(default_factory=lambda: ["fwd", "bwd"])
    report_formats: List[str] = field(
        default_factory=lambda: ["markdown", "csv", "json", "junit"]
    )
    out_dir: str = "reports"
    allow_cpu_fallback: bool = False     # run ppufn on cpu (framework self-check)
    secondary_check: bool = True         # cross-check vs cpu same-dtype golden
    dump_on_fail: bool = True
    topk: int = 8
    include_empty: bool = True
    include_large: bool = True

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Config":
        cfg = cls()
        if not d:
            return cfg
        for k, v in d.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
        return cfg


def load_config(path: Optional[str] = None) -> Config:
    """Load a YAML config file (falls back to ``configs/default.yaml``)."""
    path = path or DEFAULT_CONFIG_PATH
    data: dict = {}
    if os.path.exists(path):
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "pyyaml is required to read config files; "
                "install it or pass options via the CLI"
            ) from exc
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        # allow a top-level ``config:`` wrapper
        if isinstance(data, dict) and "config" in data and isinstance(
            data["config"], dict
        ):
            data = {**data, **data.pop("config")}
    return Config.from_dict(data)


# --------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------
def resolve_device(name: str, allow_cpu_fallback: bool = False) -> torch.device:
    """Resolve the PPU device, optionally falling back to CPU.

    The PyTorch-for-SAIL fork surfaces the 真武 PPU through the CUDA device
    layer, so ``cuda`` is the canonical PPU device string.  When no accelerator
    is present and ``allow_cpu_fallback`` is set we run on CPU -- this turns the
    suite into a golden-vs-golden self-check that validates the framework itself
    (useful in CI containers without a PPU and on dev laptops).
    """
    dev = torch.device(name)
    if dev.type == "cuda":
        if torch.cuda.is_available():
            return dev
        if allow_cpu_fallback:
            return torch.device("cpu")
        return dev  # caller will detect unavailability and skip
    return dev


def device_available(dev: torch.device) -> bool:
    if dev.type == "cuda":
        return torch.cuda.is_available()
    return True


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    case_id: str
    spec_name: str
    dtype: torch.dtype
    shape_case: inputgen.ShapeCase
    pattern: Pattern
    params: dict
    direction: str            # "fwd" | "bwd"
    seed: int
    grad_pattern: Optional[Pattern] = None

    @property
    def dtype_name(self) -> str:
        return dtype_name(self.dtype)


def _stable_seed(base_seed: int, case_id: str) -> int:
    h = zlib.crc32(case_id.encode("utf-8")) & 0xFFFF
    return (base_seed * 1000003 + h) & 0x7FFFFFFF


def _param_tag(params: dict) -> str:
    call = {k: v for k, v in params.items() if not k.startswith("_")}
    if not call:
        return "noparam"
    return "_".join(f"{k}{v}" for k, v in sorted(call.items()))


def _smoke_subset(spec: OpSpec) -> Sequence[tuple]:
    """Pick ~5 representative (shape, pattern, params) combos for smoke level."""
    shapes = spec.shapes
    # prefer one regular + one business + one special-layout shape
    chosen_shapes: List[inputgen.ShapeCase] = []
    for cat in ("regular", "business", "noncontig"):
        for sc in shapes:
            if sc.category == cat and sc not in chosen_shapes:
                chosen_shapes.append(sc)
                break
    if not chosen_shapes and shapes:
        chosen_shapes = shapes[:1]
    chosen_shapes = chosen_shapes[:3]

    patterns = spec.patterns
    chosen_patterns: List[Pattern] = []
    for p in patterns:
        if p.kind is inputgen.PatternKind.NORMAL:
            chosen_patterns.insert(0, p)
            break
    # add one "interesting" pattern (nearboundary/maskfill/extreme if present)
    for p in patterns:
        if p.kind in (
            inputgen.PatternKind.NEARBOUNDARY,
            inputgen.PatternKind.MASKFILL,
            inputgen.PatternKind.EXTREME,
            inputgen.PatternKind.CANCELLATION,
        ) and p not in chosen_patterns:
            chosen_patterns.append(p)
            break
    if not chosen_patterns and patterns:
        chosen_patterns = patterns[:1]
    chosen_patterns = chosen_patterns[:2]

    chosen_params = spec.params[:1] if spec.params else [{}]

    combos = []
    for sc in chosen_shapes:
        for pt in chosen_patterns:
            for pi, pm in enumerate(chosen_params):
                if spec.supports_case(sc, pt, pm):
                    combos.append((sc, pt, pi, pm))
    return combos[:5]


def enumerate_cases(
    specs: Sequence[OpSpec],
    cfg: Config,
    *,
    ops_filter: Optional[Sequence[str]] = None,
    category_filter: Optional[str] = None,
    dtypes_filter: Optional[Sequence[torch.dtype]] = None,
    directions: Optional[Sequence[str]] = None,
    backward_only: bool = False,
) -> List[Case]:
    """Expand the registry x config into a flat list of :class:`Case`."""
    base_seed = _resolve_base_seed(cfg.seed)
    dtypes = dtypes_filter if dtypes_filter else parse_dtypes(cfg.dtypes)
    dirs = set(directions) if directions else set(cfg.directions)
    if backward_only:
        dirs = {"bwd"}

    cases: List[Case] = []
    for spec in specs:
        if ops_filter and spec.name not in ops_filter:
            continue
        if category_filter and spec.category != category_filter:
            continue

        spec_dtypes = [d for d in dtypes if d in spec.dtypes] or []
        # FP16 is opt-in: only run it if explicitly requested via dtypes_filter
        for dtype in spec_dtypes:
            combos = (
                _smoke_subset(spec)
                if cfg.level == "smoke"
                else _full_matrix(spec, cfg)
            )
            for shape_case, pattern, pi, params in combos:
                if shape_case.category == "empty" and not cfg.include_empty:
                    continue
                if shape_case.category == "large" and not cfg.include_large:
                    continue
                run_dirs = _dirs_for_spec(spec, dirs)
                for direction in run_dirs:
                    grad_patterns = (
                        spec.grad_patterns or [inputgen.NORMAL]
                        if direction == "bwd"
                        else [None]
                    )
                    for grad_pattern in grad_patterns:
                        tag = _param_tag(params)
                        grad_tag = (
                            f"__grad-{grad_pattern.label}"
                            if grad_pattern is not None
                            else ""
                        )
                        cid = (
                            f"{spec.name}__{dtype_name(dtype)}__{shape_case.name}"
                            f"__{pattern.label}__p{pi}_{tag}__{direction}{grad_tag}"
                        )
                        cases.append(
                            Case(
                                case_id=cid,
                                spec_name=spec.name,
                                dtype=dtype,
                                shape_case=shape_case,
                                pattern=pattern,
                                params=params,
                                direction=direction,
                                seed=_stable_seed(base_seed, cid),
                                grad_pattern=grad_pattern,
                            )
                        )
    return cases


def _dirs_for_spec(spec: OpSpec, dirs: set) -> List[str]:
    out = []
    if "fwd" in dirs:
        out.append("fwd")
    if "bwd" in dirs and spec.has_backward in (
        BackwardMode.REQUIRED,
        BackwardMode.OPTIONAL,
    ):
        out.append("bwd")
    return out


def _full_matrix(spec: OpSpec, cfg: Config) -> List[tuple]:
    combos = []
    for sc in spec.shapes:
        for pt in spec.patterns:
            for pi, pm in enumerate(spec.params):
                if spec.supports_case(sc, pt, pm):
                    combos.append((sc, pt, pi, pm))
    return combos


def _resolve_base_seed(seed) -> int:
    if isinstance(seed, str) and seed.lower() == "random":
        return random.randint(0, 2**31 - 1)
    return int(seed)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
@dataclass
class CaseResult:
    case: Case
    category: str
    passed: bool
    skipped: bool = False
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    sub: List[dict] = field(default_factory=list)
    tolerance: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    notes: str = ""
    repro_cmd: str = ""
    dump_path: str = ""
    duration_s: float = 0.0

    def to_row(self) -> dict:
        c = self.case
        row = {
            "case_id": c.case_id,
            "op": c.spec_name,
            "category": self.category,
            "dtype": c.dtype_name,
            "direction": c.direction,
            "grad_pattern": c.grad_pattern.label if c.grad_pattern else "",
            "shape": tuple(c.shape_case.shape),
            "shape_category": c.shape_case.category,
            "pattern": c.pattern.label,
            "params": {k: v for k, v in c.params.items() if not k.startswith("_")},
            "seed": c.seed,
            "status": (
                "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")
            ),
            "reasons": "; ".join(self.reasons),
            "error": self.error or "",
            "duration_s": round(self.duration_s, 4),
            "dump_path": self.dump_path,
        }
        row.update({k: v for k, v in self.metrics.items()})
        return row


def _worst_metrics(subs: List[dict]) -> Dict[str, float]:
    """Reduce gating per-tensor comparisons to element-wise worst metrics.

    Informational sub-results (the CPU same-dtype cross-check) and tracebacks
    are excluded so the reported numbers reflect only what actually gates.
    """
    keys = (
        "max_abs_err",
        "max_rel_err",
        "mean_abs_err",
        "mse",
        "rmse",
        "mismatch_rate",
        "ulp_err_max",
    )
    worst: Dict[str, float] = {}
    cos_vals = []
    for s in subs:
        label = s.get("label", "")
        if label.startswith("out_vs_secondary") or label == "traceback":
            continue
        m = s.get("metrics", {})
        for k in keys:
            if k in m and m[k] == m[k]:  # skip NaN
                worst[k] = max(worst.get(k, float("-inf")), m[k])
        if "cosine_sim" in m and m["cosine_sim"] == m["cosine_sim"]:
            cos_vals.append(m["cosine_sim"])
    for k in keys:
        if k in worst and worst[k] == float("-inf"):
            del worst[k]
    if cos_vals:
        worst["cosine_sim"] = min(cos_vals)
    return worst


def _comparison_to_sub(label: str, cmp: Comparison) -> dict:
    return {
        "label": label,
        "metrics": dict(cmp.metrics),
        "reasons": list(cmp.reasons),
        "hard_ok": cmp.hard_ok,
        "top_offenders": cmp.top_offenders,
    }


def execute_case(
    case: Case,
    spec: OpSpec,
    device: torch.device,
    cfg: Config,
) -> CaseResult:
    """Run one case end-to-end and return its verdict."""
    t0 = time.time()
    tol: Tolerance = resolve_tolerance(case.dtype, spec.tolerance)
    result = CaseResult(
        case=case,
        category=spec.category,
        passed=False,
        tolerance=tol.as_dict(),
        notes=spec.notes,
    )

    if not device_available(device):
        result.skipped = True
        result.reasons.append(f"device {device} unavailable")
        return result

    try:
        _run_case_body(case, spec, device, cfg, tol, result)
    except Exception as exc:  # noqa: BLE001 - a crash IS a failing case
        import traceback

        result.error = f"{type(exc).__name__}: {exc}"
        result.passed = False
        result.reasons.append(result.error)
        result.sub.append({"label": "traceback", "traceback": traceback.format_exc()})

    result.duration_s = time.time() - t0
    result.metrics = _worst_metrics(result.sub)
    return result


def _run_case_body(case, spec, device, cfg, tol, result):
    call_params = spec.call_params(case.params)
    shapes = spec.input_shapes(case.shape_case, case.params)
    views = spec.input_views(case.shape_case, case.params)
    patterns = spec.input_patterns(case.pattern, case.params)

    want_grad = case.direction == "bwd"
    requires_grad = [want_grad] * len(shapes)

    gen = torch.Generator(device="cpu").manual_seed(case.seed)
    inputs = inputgen.build_inputs(
        patterns, shapes, views, case.dtype, device, gen
    )
    if want_grad:
        # detach() creates leaf tensors while preserving non-contiguous and
        # expanded strides; clone() would silently make those cases contiguous.
        inputs = [t.detach().requires_grad_(True) for t in inputs]

    # ---------------- forward ----------------
    ppu_out = spec.ppufn(*inputs, **call_params)
    if isinstance(ppu_out, (tuple, list)):
        ppu_out = ppu_out[0]

    inputs_fp64 = to_fp64_inputs(inputs, requires_grad)
    golden = golden_forward(
        spec.reffn,
        inputs_fp64,
        call_params,
        case.dtype,
        secondary_inputs=inputs if cfg.secondary_check else None,
        secondary_dtype=case.dtype if cfg.secondary_check else None,
    )

    cmp_out = compare(
        ppu_out, golden.out_target, tol, dtype=case.dtype,
        check_stride=spec.check_stride, empty_is_pass=spec.empty_is_pass,
        topk=cfg.topk,
    )
    result.sub.append(_comparison_to_sub("out", cmp_out))

    # secondary cross-check (informational; recorded but not gating)
    if golden.out_secondary is not None:
        cmp_sec = compare(
            ppu_out, golden.out_secondary, tol, dtype=case.dtype,
            check_shape=True, check_dtype=False, check_stride=False, topk=0,
        )
        result.sub.append(_comparison_to_sub("out_vs_secondary_cpu", cmp_sec))

    if case.direction == "fwd":
        result.passed = cmp_out.passed
        result.reasons.extend(cmp_out.reasons)
        return

    # ---------------- backward ----------------
    grad_pattern = case.grad_pattern or inputgen.NORMAL
    ggen = torch.Generator(device="cpu").manual_seed(case.seed + 1)
    grad_out = inputgen.grad_output_for(
        tuple(ppu_out.shape), case.dtype, device, ggen, grad_pattern
    )

    ppu_out.backward(grad_out)
    ppu_grads = [t.grad for t in inputs]

    golden_grads = golden_backward(golden, grad_out)
    golden_grads_target = [g.to(case.dtype) for g in golden_grads]

    bwd_ok = True
    for i, (pg, gg) in enumerate(zip(ppu_grads, golden_grads_target)):
        if pg is None:
            result.sub.append(
                _comparison_to_sub(
                    f"grad{i}",
                    Comparison(passed=False, hard_reasons=[f"grad{i} is None"]),
                )
            )
            bwd_ok = False
            continue
        cmp_g = compare(
            pg, gg, tol, dtype=case.dtype,
            check_stride=False, empty_is_pass=spec.empty_is_pass, topk=cfg.topk,
        )
        result.sub.append(_comparison_to_sub(f"grad{i}", cmp_g))
        if not cmp_g.passed:
            bwd_ok = False

    result.passed = bool(cmp_out.passed and bwd_ok)
    result.reasons.extend(cmp_out.reasons)
    for s in result.sub:
        if s["label"].startswith("grad") and s.get("reasons"):
            result.reasons.extend(f"[{s['label']}] {r}" for r in s["reasons"])


def run_cases(
    cases: Sequence[Case],
    device: torch.device,
    cfg: Config,
    *,
    on_result=None,
) -> List[CaseResult]:
    """Execute a list of cases sequentially."""
    results: List[CaseResult] = []
    for case in cases:
        spec = OP_REGISTRY[case.spec_name]
        res = execute_case(case, spec, device, cfg)
        results.append(res)
        if on_result is not None:
            on_result(res)
    return results


def run_cases_parallel(
    cases: Sequence[Case],
    device: torch.device,
    cfg: Config,
    max_workers: int = 4,
    *,
    on_result=None,
) -> List[CaseResult]:
    """Execute cases across a thread pool (torch releases the GIL on device ops).

    Results are returned in the original case order so aggregation/reports stay
    deterministic regardless of completion order.
    """
    from concurrent.futures import ThreadPoolExecutor

    cases = list(cases)
    if max_workers <= 1 or len(cases) <= 1:
        return run_cases(cases, device, cfg, on_result=on_result)

    def _one(case: Case) -> CaseResult:
        return execute_case(case, OP_REGISTRY[case.spec_name], device, cfg)

    results: List[CaseResult] = [None] * len(cases)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, c): i for i, c in enumerate(cases)}
        for fut, i in futures.items():
            res = fut.result()
            results[i] = res
            if on_result is not None:
                on_result(res)
    return results


__all__ = [
    "Config",
    "load_config",
    "resolve_device",
    "device_available",
    "Case",
    "CaseResult",
    "enumerate_cases",
    "execute_case",
    "run_cases",
    "run_cases_parallel",
    "all_specs",
]
