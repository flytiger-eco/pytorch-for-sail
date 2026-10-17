"""Failure-scene dump + reproducible command generation (§5.5 / §7).

Because every case is generated from a deterministic per-case seed, the dump
step simply *re-runs* the case and persists everything needed to triage it
offline: the input tensor(s), the PPU output, the golden output, the diff and a
ready-to-paste repro command.
"""

from __future__ import annotations

import json
import os
import shlex
from typing import Optional

import torch

from . import inputgen
from .opspec import OpSpec
from .reference import golden_forward, to_fp64_inputs
from .runner import Case, Config
from .tolerance import dtype_name


def repro_command(case: Case, cfg: Optional[Config] = None) -> str:
    """A copy-pasteable command that re-runs exactly this case."""
    c = case
    params = {k: v for k, v in c.params.items() if not k.startswith("_")}
    if c.dtype_ref is not None:
        # cross-dtype case: reproduce both dtypes and re-enable the mode.
        dt = f"{dtype_name(c.dtype)},{dtype_name(c.dtype_ref)}"
        mode_flag = " --cross-dtype"
    else:
        dt = dtype_name(c.dtype)
        mode_flag = ""
    base = (
        f"python run.py --ops {shlex.quote(c.spec_name)} --dtypes {dt}"
        f"{mode_flag} "
        f"--level full --case-id {shlex.quote(c.case_id)} "
        f"--case-seed {c.seed}"
    )
    if cfg is not None:
        if cfg.device != "cuda":
            base += f" --device {shlex.quote(cfg.device)}"
        if cfg.allow_cpu_fallback:
            base += " --allow-cpu-fallback"
        if not cfg.secondary_check:
            base += " --no-secondary"
    if params:
        base += f"  # params={params}"
    base += f"  # shape={tuple(c.shape_case.shape)} pattern={c.pattern.label}"
    return base


def _safe_name(case: Case) -> str:
    return case.case_id.replace("/", "_").replace(" ", "_")


def _golden_artifacts(
    case, spec, device, cfg,
    call_params, shapes, views, patterns, want_grad,
) -> dict:
    """Re-run the golden path and collect PPU-vs-fp64 triage artifacts."""
    gen = torch.Generator(device="cpu").manual_seed(case.seed)
    inputs = inputgen.build_inputs(
        patterns, shapes, views, case.dtype, device, gen
    )
    if want_grad:
        inputs = [t.detach().requires_grad_(True) for t in inputs]

    artifacts = {"inputs": [t.detach().cpu() for t in inputs]}
    try:
        ppu_out = spec.ppufn(*inputs, **call_params)
        if isinstance(ppu_out, (tuple, list)):
            ppu_out = ppu_out[0]
        artifacts["ppu_out"] = ppu_out.detach().cpu()

        inputs_fp64 = to_fp64_inputs(inputs, [want_grad] * len(inputs))
        golden = golden_forward(
            spec.reffn,
            inputs_fp64,
            call_params,
            case.dtype,
            secondary_inputs=inputs if cfg.secondary_check else None,
            secondary_dtype=case.dtype if cfg.secondary_check else None,
        )
        artifacts["golden_out"] = golden.out_target.cpu()
        artifacts["golden_out_fp64"] = golden.out_fp64.detach().cpu()
        if golden.out_secondary is not None:
            artifacts["golden_out_secondary_cpu"] = golden.out_secondary.cpu()

        artifacts["diff"] = (
            artifacts["ppu_out"].to(torch.float64)
            - artifacts["golden_out"].to(torch.float64)
        )

        if want_grad:
            grad_pattern = case.grad_pattern or inputgen.NORMAL
            from .reference import golden_backward

            ggen = torch.Generator(device="cpu").manual_seed(case.seed + 1)
            grad_out = inputgen.grad_output_for(
                tuple(ppu_out.shape), case.dtype, device, ggen, grad_pattern
            )
            artifacts["grad_output"] = grad_out.detach().cpu()
            ppu_out.backward(grad_out)
            artifacts["ppu_grads"] = [
                (t.grad.detach().cpu() if t.grad is not None else None)
                for t in inputs
            ]
            ggrads = golden_backward(golden, grad_out)
            artifacts["golden_grads"] = [g.to(case.dtype).cpu() for g in ggrads]
    except Exception as exc:  # noqa: BLE001
        artifacts["error"] = f"{type(exc).__name__}: {exc}"
    return artifacts


def _cross_dtype_artifacts(
    case, spec, device,
    call_params, shapes, views, patterns, want_grad,
) -> dict:
    """Re-run both PPU dtypes and collect low-vs-high triage artifacts.

    Mirrors :func:`core.runner._run_cross_dtype_body`: the input is built once in
    the high dtype then rounded down, so ``ppu_out_low`` / ``ppu_out_high`` are
    directly comparable and ``diff`` is their element-wise difference.
    """
    lo_dtype = case.dtype
    hi_dtype = case.dtype_ref

    gen = torch.Generator(device="cpu").manual_seed(case.seed)
    inputs_hi_plain = inputgen.build_inputs(
        patterns, shapes, views, hi_dtype, device, gen
    )
    inputs_lo_plain = [t.to(lo_dtype) for t in inputs_hi_plain]
    if want_grad:
        inputs_hi = [t.detach().requires_grad_(True) for t in inputs_hi_plain]
        inputs_lo = [t.detach().requires_grad_(True) for t in inputs_lo_plain]
    else:
        inputs_hi = inputs_hi_plain
        inputs_lo = inputs_lo_plain

    artifacts = {
        "inputs_low": [t.detach().cpu() for t in inputs_lo_plain],
        "inputs_high": [t.detach().cpu() for t in inputs_hi_plain],
    }
    try:
        out_hi = spec.ppufn(*inputs_hi, **call_params)
        if isinstance(out_hi, (tuple, list)):
            out_hi = out_hi[0]
        out_lo = spec.ppufn(*inputs_lo, **call_params)
        if isinstance(out_lo, (tuple, list)):
            out_lo = out_lo[0]
        artifacts["ppu_out_low"] = out_lo.detach().cpu()
        artifacts["ppu_out_high"] = out_hi.detach().cpu()
        artifacts["diff"] = (
            artifacts["ppu_out_low"].to(torch.float64)
            - artifacts["ppu_out_high"].to(torch.float64)
        )

        if want_grad:
            grad_pattern = case.grad_pattern or inputgen.NORMAL
            ggen = torch.Generator(device="cpu").manual_seed(case.seed + 1)
            grad_out_hi = inputgen.grad_output_for(
                tuple(out_hi.shape), hi_dtype, device, ggen, grad_pattern
            )
            grad_out_lo = grad_out_hi.to(lo_dtype)
            artifacts["grad_output_high"] = grad_out_hi.detach().cpu()
            artifacts["grad_output_low"] = grad_out_lo.detach().cpu()
            out_hi.backward(grad_out_hi)
            out_lo.backward(grad_out_lo)
            artifacts["ppu_grads_low"] = [
                (t.grad.detach().cpu() if t.grad is not None else None)
                for t in inputs_lo
            ]
            artifacts["ppu_grads_high"] = [
                (t.grad.detach().cpu() if t.grad is not None else None)
                for t in inputs_hi
            ]
    except Exception as exc:  # noqa: BLE001
        artifacts["error"] = f"{type(exc).__name__}: {exc}"
    return artifacts


def dump_case(
    case: Case,
    spec: OpSpec,
    device: torch.device,
    cfg: Config,
    out_root: str,
) -> str:
    """Re-run ``case``, persist tensors + metadata; returns the dump directory."""
    d = os.path.join(out_root, "dumps", _safe_name(case))
    os.makedirs(d, exist_ok=True)

    call_params = spec.call_params(case.params)
    shapes = spec.input_shapes(case.shape_case, case.params)
    views = spec.input_views(case.shape_case, case.params)
    patterns = spec.input_patterns(case.pattern, case.params)
    want_grad = case.direction == "bwd"

    if case.dtype_ref is not None:
        artifacts = _cross_dtype_artifacts(
            case, spec, device, call_params, shapes, views, patterns, want_grad
        )
    else:
        artifacts = _golden_artifacts(
            case, spec, device, cfg,
            call_params, shapes, views, patterns, want_grad,
        )

    torch.save(artifacts, os.path.join(d, "case.pt"))

    meta = {
        "case_id": case.case_id,
        "op": case.spec_name,
        "category": spec.category,
        "dtype": dtype_name(case.dtype),
        "compare_mode": "cross-dtype" if case.dtype_ref is not None else "golden",
        "dtype_ref": (
            dtype_name(case.dtype_ref) if case.dtype_ref is not None else ""
        ),
        "direction": case.direction,
        "grad_pattern": case.grad_pattern.label if case.grad_pattern else "",
        "shape": list(case.shape_case.shape),
        "shape_category": case.shape_case.category,
        "view": case.shape_case.view,
        "pattern": case.pattern.label,
        "pattern_kind": case.pattern.kind.value,
        "params": {k: v for k, v in case.params.items()},
        "seed": case.seed,
        "device": str(device),
        "notes": spec.notes,
        "repro_cmd": repro_command(case, cfg),
        "error": artifacts.get("error", ""),
        "artifact_keys": [k for k in artifacts if k != "error"],
    }
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # standalone repro script. Use the actual framework root so custom --out
    # directories do not change where run.py is resolved from.
    framework_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(d, "repro.sh"), "w") as fh:
        fh.write("#!/usr/bin/env bash\n")
        fh.write("# Reproduce this failing case standalone.\n")
        fh.write("set -euo pipefail\n")
        fh.write(f"cd {shlex.quote(framework_root)}\n")
        fh.write(repro_command(case, cfg).split("  #")[0].strip() + "\n")
    os.chmod(os.path.join(d, "repro.sh"), 0o755)

    return d
