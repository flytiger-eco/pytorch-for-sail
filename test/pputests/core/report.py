"""Report generation and environment snapshot (§7 of the spec).

Produces, into the output directory:

* ``summary.csv``      -- one row per case (machine-readable flat table)
* ``results.json``     -- full structured results incl. sub-comparisons + env
* ``junit.xml``        -- CI-consumable JUnit format
* ``summary.md``       -- human Markdown: pivot table + failure detail
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import subprocess
from typing import Dict, List, Optional, Sequence

from .runner import CaseResult
from .tolerance import dtype_name

_STATUS_ORDER = {"FAIL": 0, "SKIP": 1, "PASS": 2}

# Metrics surfaced in the pivot table (max across cases of each op cell).
_PIVOT_METRICS = (
    "max_abs_err",
    "max_rel_err",
    "cosine_sim",
    "mismatch_rate",
)


# --------------------------------------------------------------------------
# environment snapshot
# --------------------------------------------------------------------------
def environment_snapshot() -> Dict[str, object]:
    """Collect PPU driver / PyTorch / commit / timestamp metadata."""
    env: Dict[str, object] = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["torch_git_version"] = getattr(torch.version, "git_version", "")
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                env["device_name"] = torch.cuda.get_device_name(0)
                env["device_count"] = torch.cuda.device_count()
            except Exception:  # pragma: no cover
                pass
    except Exception as exc:  # pragma: no cover
        env["torch_error"] = str(exc)

    env["commit"] = _git_rev_parse()
    env["ppu_smi"] = _ppu_smi()
    env["ppu_sdk"] = os.environ.get("PPU_SDK", "")
    return env


def _git_rev_parse() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # pragma: no cover
        return ""


def _ppu_smi() -> str:
    """Best-effort PPU driver/firmware version via ``ppu-smi``."""
    try:
        out = subprocess.run(
            ["ppu-smi"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            first = [ln for ln in out.stdout.splitlines() if ln.strip()][:3]
            return " | ".join(first)
    except Exception:  # pragma: no cover
        pass
    return ""


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
def write_reports(
    results: Sequence[CaseResult],
    out_dir: str,
    formats: Sequence[str],
    env: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    """Write the requested report formats; returns {format: path}."""
    os.makedirs(out_dir, exist_ok=True)
    env = env if env is not None else environment_snapshot()
    written: Dict[str, str] = {}
    fmts = {f.lower() for f in formats}

    if "csv" in fmts:
        written["csv"] = _write_csv(results, out_dir)
    if "json" in fmts:
        written["json"] = _write_json(results, out_dir, env)
    if "junit" in fmts or "junit-xml" in fmts:
        written["junit"] = _write_junit(results, out_dir, env)
    if "markdown" in fmts or "md" in fmts:
        written["markdown"] = _write_markdown(results, out_dir, env)
    return written


def _rows(results: Sequence[CaseResult]) -> List[dict]:
    return [r.to_row() for r in results]


def _write_csv(results: Sequence[CaseResult], out_dir: str) -> str:
    path = os.path.join(out_dir, "summary.csv")
    rows = _rows(results)
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return path


def _write_json(
    results: Sequence[CaseResult], out_dir: str, env: Dict[str, object]
) -> str:
    path = os.path.join(out_dir, "results.json")
    payload = {
        "environment": env,
        "totals": _totals(results),
        "cases": [_json_case(r) for r in results],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _json_case(r: CaseResult) -> dict:
    row = r.to_row()
    row["sub"] = r.sub
    row["tolerance"] = r.tolerance
    row["notes"] = r.notes
    row["repro_cmd"] = r.repro_cmd
    return row


def _totals(results: Sequence[CaseResult]) -> Dict[str, int]:
    tot = {"total": len(results), "pass": 0, "fail": 0, "skip": 0}
    for r in results:
        if r.skipped:
            tot["skip"] += 1
        elif r.passed:
            tot["pass"] += 1
        else:
            tot["fail"] += 1
    return tot


def _write_junit(
    results: Sequence[CaseResult], out_dir: str, env: Dict[str, object]
) -> str:
    path = os.path.join(out_dir, "junit.xml")
    tot = _totals(results)
    total_time = sum(r.duration_s for r in results)
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<testsuites tests="{tot["total"]}" failures="{tot["fail"]}" '
        f'skipped="{tot["skip"]}" time="{total_time:.3f}">',
        f'  <testsuite name="ppu_accuracy" tests="{tot["total"]}" '
        f'failures="{tot["fail"]}" skipped="{tot["skip"]}" '
        f'time="{total_time:.3f}" timestamp="{env.get("timestamp", "")}">',
    ]
    for r in results:
        c = r.case
        cls = f"{c.spec_name}.{c.dtype_label}"
        name = c.case_id
        lines.append(
            f'    <testcase classname="{_esc(cls)}" name="{_esc(name)}" '
            f'time="{r.duration_s:.4f}">'
        )
        if r.skipped:
            lines.append(
                f'      <skipped message="{_esc("; ".join(r.reasons))}"/>'
            )
        elif not r.passed:
            msg = "; ".join(r.reasons) or (r.error or "failed")
            lines.append(f'      <failure message="{_esc(msg)}">')
            lines.append(f"{_esc(_failure_text(r))}")
            lines.append("      </failure>")
        lines.append("    </testcase>")
    lines.append("  </testsuite>")
    lines.append("</testsuites>")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _failure_text(r: CaseResult) -> str:
    parts = [f"case_id: {r.case.case_id}", f"status: FAIL"]
    if r.error:
        parts.append(f"error: {r.error}")
    parts.append("reasons: " + "; ".join(r.reasons))
    for k in _PIVOT_METRICS:
        if k in r.metrics:
            parts.append(f"{k}: {r.metrics[k]:.6g}")
    parts.append(f"tolerance: {r.tolerance}")
    if r.dump_path:
        parts.append(f"dump: {r.dump_path}")
    return "\n".join(parts)


def _write_markdown(
    results: Sequence[CaseResult], out_dir: str, env: Dict[str, object]
) -> str:
    path = os.path.join(out_dir, "summary.md")
    tot = _totals(results)
    lines: List[str] = []
    lines.append("# PPU 算子精度对比测试报告")
    lines.append("")
    lines.append(f"- 生成时间: {env.get('timestamp','')}")
    lines.append(f"- PyTorch 版本: {env.get('torch_version','?')}")
    lines.append(f"- commit: {env.get('commit','?')}")
    if env.get("ppu_smi"):
        lines.append(f"- PPU: {env.get('ppu_smi')}")
    lines.append(
        f"- 汇总: total={tot['total']} PASS={tot['pass']} "
        f"FAIL={tot['fail']} SKIP={tot['skip']}"
    )
    if any(r.case.dtype_ref is not None for r in results):
        lines.append(
            "- 模式: 跨 dtype 对比 (无 fp64 golden)。`ppu=` 为低精度 PPU 结果，"
            "`ref=` 为高精度 PPU 结果；相对误差 = |低精度 − 高精度| / |高精度|。"
        )
    lines.append("")

    # ---- pivot: op x dtype x direction ----------------------------------
    lines.append("## 汇总表 (算子 × dtype × 方向)")
    lines.append("")
    header = ["op", "dtype", "dir", "cases", "PASS", "FAIL", "SKIP"]
    header += list(_PIVOT_METRICS)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    cells: Dict[tuple, List[CaseResult]] = {}
    for r in results:
        key = (r.case.spec_name, r.case.dtype_label, r.case.direction)
        cells.setdefault(key, []).append(r)

    for key in sorted(cells):
        rs = cells[key]
        op, dt, direction = key
        npass = sum(1 for x in rs if x.passed and not x.skipped)
        nfail = sum(1 for x in rs if not x.passed and not x.skipped)
        nskip = sum(1 for x in rs if x.skipped)
        agg = _aggregate_metrics(rs)
        row = [
            op, dt, direction, str(len(rs)),
            str(npass), str(nfail), str(nskip),
        ]
        for m in _PIVOT_METRICS:
            row.append(f"{agg[m]:.4g}" if m in agg else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- failure detail --------------------------------------------------
    failures = [r for r in results if not r.passed and not r.skipped]
    lines.append(f"## 失败明细 ({len(failures)})")
    lines.append("")
    if not failures:
        lines.append("_无失败用例_ 🎉")
    for r in sorted(failures, key=lambda x: x.case.case_id):
        c = r.case
        lines.append(f"### `{c.case_id}`")
        lines.append(f"- shape: {tuple(c.shape_case.shape)} ({c.shape_case.category})")
        lines.append(f"- pattern: {c.pattern.label} | params: "
                     f"{ {k:v for k,v in c.params.items() if not k.startswith('_')} }")
        lines.append(f"- seed: {c.seed}")
        if r.error:
            lines.append(f"- error: `{r.error}`")
        for reason in r.reasons:
            lines.append(f"- ❌ {reason}")
        lines.append(f"- tolerance: {r.tolerance}")
        for s in r.sub:
            if s.get("top_offenders"):
                lines.append(f"- top offenders ({s['label']}):")
                for idx, ov, rv, ae in s["top_offenders"][:5]:
                    lines.append(f"    - flat_idx={idx} ppu={ov:.6g} ref={rv:.6g} "
                                 f"abs_err={ae:.3e}")
        if r.dump_path:
            lines.append(f"- dump: `{r.dump_path}`")
        if r.repro_cmd:
            lines.append(f"- repro: `{r.repro_cmd}`")
        lines.append("")

    # ---- notes / relaxations --------------------------------------------
    noted = {
        r.case.spec_name: r.notes for r in results if r.notes
    }
    if noted:
        lines.append("## 算子备注 / 阈值放宽理由")
        lines.append("")
        for op in sorted(noted):
            lines.append(f"- **{op}**: {noted[op]}")
        lines.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def _aggregate_metrics(rs: Sequence[CaseResult]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for m in _PIVOT_METRICS:
        vals = [r.metrics[m] for r in rs if m in r.metrics and r.metrics[m] == r.metrics[m]]
        if not vals:
            continue
        if m == "cosine_sim":
            agg[m] = min(vals)
        else:
            agg[m] = max(vals)
    return agg
