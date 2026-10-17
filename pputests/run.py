#!/usr/bin/env python3
"""CLI entry point for the PPU operator accuracy framework (§6.4 of the spec).

Examples::

    python run.py --ops relu,softmax --dtypes bf16,fp32 --level full --seed 0
    python run.py --category activation --backward-only
    python run.py --all --report markdown,junit --out reports/
    python run.py --all --level smoke --allow-cpu-fallback   # no PPU needed
    python run.py --list                                     # show registered ops

Exit code is non-zero when any case FAILs (skips do not fail the run), so the
script drops straight into CI.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ops  # noqa: E402,F401  (registers operators)
from core import report as report_mod  # noqa: E402
from core import runner  # noqa: E402
from core.dump import dump_case, repro_command  # noqa: E402
from core.opspec import OP_REGISTRY, all_specs, resolve_spec_names  # noqa: E402
from core.runner import Config, enumerate_cases, run_cases_parallel  # noqa: E402
from core.tolerance import dtype_name, parse_dtypes  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PPU operator accuracy comparison runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ops",
        default=None,
        help="comma list of exact operator names (e.g. relu,softmax,sum)",
    )
    p.add_argument("--category", default=None, help="only this category")
    p.add_argument("--all", action="store_true", help="run every registered op")
    p.add_argument("--dtypes", default=None, help="bf16,fp32[,fp16]")
    p.add_argument("--level", default=None, choices=["smoke", "full"])
    p.add_argument("--seed", default=None, help="int or 'random'")
    p.add_argument("--device", default=None, help="PPU device string (cuda)")
    p.add_argument("--direction", default=None, help="fwd,bwd")
    p.add_argument("--backward-only", action="store_true")
    p.add_argument(
        "--case-id",
        default=None,
        help="run one exact generated case ID (used by failure repro scripts)",
    )
    p.add_argument(
        "--case-seed",
        type=int,
        default=None,
        help="override the selected case's derived seed; requires --case-id",
    )
    p.add_argument("--report", default=None,
                   help="comma list: markdown,csv,json,junit")
    p.add_argument("--out", default=None, help="report output directory")
    p.add_argument("--config", default=None, help="path to a configs/*.yaml")
    p.add_argument("--allow-cpu-fallback", action="store_true",
                   help="run ppufn on CPU when no PPU is present (self-check)")
    p.add_argument("--no-dump", action="store_true", help="disable failure dumps")
    p.add_argument("--no-secondary", action="store_true",
                   help="disable the CPU same-dtype cross-check")
    p.add_argument("-n", "--jobs", type=int, default=1,
                   help="parallel worker threads for case execution (full mode)")
    p.add_argument("--list", action="store_true", help="list registered ops & exit")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def resolve_op_names(tokens: str) -> list:
    """Resolve comma-separated exact operator names from the registry."""
    if not tokens:
        return []
    return resolve_spec_names(tokens.split(","))


def cfg_from_args(args: argparse.Namespace) -> Config:
    cfg = runner.load_config(args.config)
    if args.level:
        cfg.level = args.level
    if args.dtypes:
        cfg.dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    if args.seed is not None:
        cfg.seed = args.seed if args.seed == "random" else int(args.seed)
    if args.device:
        cfg.device = args.device
    if args.out:
        cfg.out_dir = args.out
    if args.report:
        cfg.report_formats = [f.strip() for f in args.report.split(",") if f.strip()]
    if args.allow_cpu_fallback:
        cfg.allow_cpu_fallback = True
    if args.no_dump:
        cfg.dump_on_fail = False
    if args.no_secondary:
        cfg.secondary_check = False
    return cfg


def _print_ops():
    print(f"registered operators ({len(OP_REGISTRY)}):")
    for spec in all_specs():
        dts = ",".join(dtype_name(d) for d in spec.dtypes)
        print(
            f"  {spec.name:<16} cat={spec.category:<14} "
            f"dtypes=[{dts}] bwd={spec.has_backward.value} "
            f"shapes={len(spec.shapes)} patterns={len(spec.patterns)} "
            f"params={len(spec.params)}"
        )


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.list:
        _print_ops()
        return 0

    cfg = cfg_from_args(args)
    if args.case_seed is not None and not args.case_id:
        print("[ppu-accuracy] --case-seed requires --case-id", file=sys.stderr)
        return 2
    device = runner.resolve_device(cfg.device, cfg.allow_cpu_fallback)

    if not args.quiet:
        avail = runner.device_available(device)
        print(f"[ppu-accuracy] device={device} available={avail} "
              f"level={cfg.level} dtypes={cfg.dtypes}")
        if not avail:
            print("[ppu-accuracy] PPU unavailable; all cases will be SKIPPED. "
                  "Use --allow-cpu-fallback for a golden self-check.")

    # ---- selection ----
    specs = all_specs()
    ops_filter = None
    if args.ops:
        try:
            ops_filter = resolve_op_names(args.ops)
        except ValueError as exc:
            print(f"[ppu-accuracy] {exc}", file=sys.stderr)
            return 2
    elif not args.all and not args.category:
        # default: everything (equivalent to --all)
        ops_filter = None

    directions = None
    if args.direction:
        directions = [d.strip() for d in args.direction.split(",") if d.strip()]

    cases = enumerate_cases(
        specs, cfg,
        ops_filter=ops_filter,
        category_filter=args.category,
        dtypes_filter=parse_dtypes(cfg.dtypes),
        directions=directions,
        backward_only=args.backward_only,
    )
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
        if cases and args.case_seed is not None:
            cases = [replace(cases[0], seed=args.case_seed)]

    if not cases:
        print("[ppu-accuracy] no cases matched the selection", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"[ppu-accuracy] {len(cases)} cases to run")

    # ---- execute ----
    def on_result(res):
        if args.quiet:
            return
        status = "SKIP" if res.skipped else ("PASS" if res.passed else "FAIL")
        mark = {"PASS": ".", "FAIL": "F", "SKIP": "s"}[status]
        sys.stdout.write(mark)
        sys.stdout.flush()

    results = run_cases_parallel(
        cases, device, cfg, max_workers=args.jobs, on_result=on_result
    )
    if not args.quiet:
        print()

    # ---- dump failures ----
    if cfg.dump_on_fail:
        for res in results:
            if res.passed or res.skipped:
                continue
            res.repro_cmd = repro_command(res.case, cfg)
            try:
                res.dump_path = dump_case(
                    res.case, OP_REGISTRY[res.case.spec_name], device, cfg, cfg.out_dir
                )
            except Exception as exc:  # noqa: BLE001
                res.dump_path = f"<dump failed: {exc}>"

    # ---- report ----
    env = report_mod.environment_snapshot()
    paths = report_mod.write_reports(results, cfg.out_dir, cfg.report_formats, env)

    tot = report_mod._totals(results)
    if not args.quiet:
        print(f"[ppu-accuracy] total={tot['total']} PASS={tot['pass']} "
              f"FAIL={tot['fail']} SKIP={tot['skip']}")
        print(f"[ppu-accuracy] reports: {paths}")
        failures = [r for r in results if not r.passed and not r.skipped]
        for r in failures[:20]:
            print(f"  FAIL {r.case.case_id}: {'; '.join(r.reasons)[:160]}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more failures")

    return 1 if tot["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
