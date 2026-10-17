"""pytest integration: options, fixtures, result collection and report hooks.

Run examples::

    pytest pputests/tests/test_ops.py -k "softmax and bf16"
    pytest pputests/tests/test_ops.py --ppu-level full --ppu-dtypes bf16,fp32
    pytest pputests/tests/test_ops.py --ppu-allow-cpu-fallback   # no PPU needed
"""

from __future__ import annotations

import os
import sys

import pytest

# Make ``core`` / ``ops`` importable regardless of the invocation directory.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ops  # noqa: E402,F401  (import triggers @register side-effects)
from core import report as report_mod  # noqa: E402
from core import runner  # noqa: E402
from core.opspec import all_specs, resolve_spec_names  # noqa: E402
from core.runner import Config, enumerate_cases  # noqa: E402
from core.tolerance import parse_dtypes  # noqa: E402


def pytest_addoption(parser):
    g = parser.getgroup("ppu-accuracy", "PPU operator accuracy framework")
    g.addoption("--ppu-config", action="store", default=None,
                help="path to a YAML config (configs/*.yaml)")
    g.addoption("--ppu-level", action="store", default=None,
                choices=["smoke", "full"], help="case-matrix level")
    g.addoption("--ppu-dtypes", action="store", default=None,
                help="comma list: bf16,fp32,fp16")
    g.addoption("--ppu-ops", action="store", default=None,
                help="comma list of op names to run")
    g.addoption("--ppu-category", action="store", default=None,
                help="only run this category")
    g.addoption("--ppu-direction", action="store", default=None,
                help="fwd,bwd")
    g.addoption("--ppu-backward-only", action="store_true", default=False)
    g.addoption("--ppu-device", action="store", default=None,
                help="PPU device string (default cuda)")
    g.addoption("--ppu-seed", action="store", default=None,
                help="int or 'random'")
    g.addoption("--ppu-out", action="store", default=None,
                help="report output directory")
    g.addoption("--ppu-report", action="store", default=None,
                help="comma list: markdown,csv,json,junit")
    g.addoption("--ppu-allow-cpu-fallback", action="store_true", default=False,
                help="run ppufn on CPU when no PPU is present (self-check)")
    g.addoption("--ppu-no-dump", action="store_true", default=False,
                help="disable failure dumps")
    g.addoption("--ppu-cross-dtype", action="store_true", default=False,
                help="skip the fp64 golden and compare two PPU dtypes "
                     "head-to-head (requires exactly two --ppu-dtypes)")


def _cfg_from_pytest(pytestconfig) -> Config:
    cfg = runner.load_config(pytestconfig.getoption("--ppu-config"))
    o = pytestconfig.option
    if o.ppu_level:
        cfg.level = o.ppu_level
    if o.ppu_dtypes:
        cfg.dtypes = [d.strip() for d in o.ppu_dtypes.split(",") if d.strip()]
    if o.ppu_device:
        cfg.device = o.ppu_device
    if o.ppu_seed is not None:
        cfg.seed = o.ppu_seed
    if o.ppu_out:
        cfg.out_dir = o.ppu_out
    if o.ppu_report:
        cfg.report_formats = [f.strip() for f in o.ppu_report.split(",") if f.strip()]
    if o.ppu_allow_cpu_fallback:
        cfg.allow_cpu_fallback = True
    if o.ppu_no_dump:
        cfg.dump_on_fail = False
    if o.ppu_cross_dtype:
        cfg.cross_dtype = True
    return cfg


def _sel_from_pytest(pytestconfig):
    o = pytestconfig.option
    ops_filter = None
    if o.ppu_ops:
        try:
            ops_filter = resolve_spec_names(o.ppu_ops.split(","))
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc
    directions = None
    if o.ppu_direction:
        directions = [x.strip() for x in o.ppu_direction.split(",") if x.strip()]
    return ops_filter, o.ppu_category, directions, o.ppu_backward_only


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    cfg = _cfg_from_pytest(metafunc.config)
    ops_filter, category, directions, bwd_only = _sel_from_pytest(metafunc.config)
    dtypes = parse_dtypes(cfg.dtypes)
    cases = enumerate_cases(
        all_specs(), cfg,
        ops_filter=ops_filter, category_filter=category,
        dtypes_filter=dtypes, directions=directions, backward_only=bwd_only,
    )
    metafunc.parametrize(
        "case", cases, ids=[c.case_id for c in cases], scope="function"
    )


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ppu_ctx(request):
    """Resolve (device, config) once per session."""
    cfg = _cfg_from_pytest(request.config)
    device = runner.resolve_device(cfg.device, cfg.allow_cpu_fallback)
    return device, cfg


@pytest.fixture(scope="session")
def result_collector():
    return []


@pytest.fixture(scope="session")
def ppu_env_snapshot():
    return report_mod.environment_snapshot()


def pytest_sessionfinish(session, exitstatus):
    """Write the aggregated report from everything collected during the run."""
    collector = getattr(session, "_ppu_results", None)
    if not collector:
        return
    cfg = getattr(session, "_ppu_cfg", None) or Config()
    env = getattr(session, "_ppu_env", None)

    # Under pytest-xdist each worker has its own session; write to a
    # worker-scoped subdir so parallel runs don't clobber each other. CI can
    # then consume reports/workers/*/junit.xml (glob).
    out_dir = cfg.out_dir
    workerinput = getattr(session.config, "workerinput", None)
    if workerinput:
        wid = workerinput.get("workerid", "gw").replace("/", "_")
        out_dir = os.path.join(cfg.out_dir, "workers", wid)

    try:
        paths = report_mod.write_reports(
            collector, out_dir, cfg.report_formats, env
        )
        writer = session.config.get_terminal_writer()
        writer.line("")
        writer.line(f"[ppu-accuracy] reports written: {paths}")
    except Exception as exc:  # pragma: no cover
        session.config.get_terminal_writer().line(
            f"[ppu-accuracy] report generation failed: {exc}"
        )


@pytest.fixture(autouse=True)
def _ppu_capture(request, result_collector, ppu_ctx, ppu_env_snapshot):
    """Stash results/config/env on the session for pytest_sessionfinish."""
    request.session._ppu_results = result_collector
    request.session._ppu_cfg = ppu_ctx[1]
    request.session._ppu_env = ppu_env_snapshot
    yield
