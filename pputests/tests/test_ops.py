"""Parametrized driver: iterate ``OP_REGISTRY`` x config and assert accuracy.

The actual case list is injected by ``conftest.pytest_generate_tests`` (which
reads the ``--ppu-*`` options), so this file stays tiny and purely declarative.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.dump import dump_case, repro_command  # noqa: E402
from core.opspec import OP_REGISTRY  # noqa: E402
from core.runner import execute_case  # noqa: E402


def test_op_accuracy(case, ppu_ctx, result_collector):
    device, cfg = ppu_ctx
    spec = OP_REGISTRY[case.spec_name]

    result = execute_case(case, spec, device, cfg)
    result.repro_cmd = repro_command(case, cfg)

    if not result.passed and not result.skipped and cfg.dump_on_fail:
        try:
            result.dump_path = dump_case(case, spec, device, cfg, cfg.out_dir)
        except Exception as exc:  # pragma: no cover - dump is best-effort
            result.dump_path = f"<dump failed: {exc}>"

    result_collector.append(result)

    if result.skipped:
        pytest.skip("; ".join(result.reasons) or "skipped")

    assert result.passed, _fmt_failure(result)


def _fmt_failure(result) -> str:
    lines = [
        f"case_id: {result.case.case_id}",
        f"shape:   {tuple(result.case.shape_case.shape)} "
        f"({result.case.shape_case.category})",
        f"pattern: {result.case.pattern.label}",
        f"seed:    {result.case.seed}",
    ]
    if result.error:
        lines.append(f"error:   {result.error}")
    lines.append("reasons: " + "; ".join(result.reasons))
    lines.append(f"metrics: {result.metrics}")
    lines.append(f"tol:     {result.tolerance}")
    if result.dump_path:
        lines.append(f"dump:    {result.dump_path}")
    if result.repro_cmd:
        lines.append(f"repro:   {result.repro_cmd}")
    return "\n".join(lines)
