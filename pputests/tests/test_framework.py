"""Focused regression tests for the accuracy framework itself."""

from __future__ import annotations

import os
import shlex
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ops  # noqa: E402,F401
from core import inputgen  # noqa: E402
from core.dump import dump_case, repro_command  # noqa: E402
from core.opspec import BackwardMode, OpSpec, get_spec  # noqa: E402
from core.runner import Case, Config, enumerate_cases, execute_case  # noqa: E402
from core.tolerance import FP32  # noqa: E402


def _case(spec_name="identity", *, shape=(0,), view=None, grad_pattern=None):
    shape_case = inputgen.ShapeCase("test", shape, view=view)
    return Case(
        case_id=f"{spec_name}__fp32__test__normal__p0_noparam__bwd",
        spec_name=spec_name,
        dtype=FP32,
        shape_case=shape_case,
        pattern=inputgen.NORMAL,
        params={},
        direction="bwd",
        seed=123,
        grad_pattern=grad_pattern or inputgen.NORMAL,
    )


def _identity_spec(ppufn=None):
    ppufn = ppufn or (lambda x: x)
    return OpSpec(
        name="identity",
        category="framework_test",
        dtypes=[FP32],
        shapes=[inputgen.ShapeCase("empty", (0,))],
        patterns=[inputgen.NORMAL],
        params=[{}],
        reffn=lambda x: x,
        ppufn=ppufn,
        has_backward=BackwardMode.REQUIRED,
    )


def test_nearboundary_ulp_uses_representable_step():
    assert inputgen._ulp_at(1.0, torch.float32) == torch.finfo(torch.float32).eps
    zero = torch.tensor(0.0, dtype=torch.float32)
    positive = torch.nextafter(zero, torch.tensor(float("inf")))
    negative = torch.nextafter(zero, torch.tensor(float("-inf")))
    pattern = inputgen.NEARBOUNDARY([0.0], ulp_span=2)
    values = inputgen._nearboundary_values(
        pattern, 6, torch.float32, torch.Generator().manual_seed(0)
    )
    assert positive in values
    assert negative in values


@pytest.mark.parametrize(
    "pattern",
    [inputgen.EXTREME_VALUES(0.5), inputgen.MASKFILL(float("-inf"), 0.5)],
)
def test_special_patterns_keep_target_device(pattern):
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        gen = torch.Generator(device="cpu").manual_seed(0)
        out = inputgen.generate_input(pattern, (16,), FP32, device, gen)
        assert out.device.type == device.type


def test_cancellation_preserves_effective_noncontiguous_shape():
    gen = torch.Generator(device="cpu").manual_seed(0)
    a, b = inputgen.build_inputs(
        [inputgen.CANCELLATION, inputgen.CANCELLATION],
        [(2, 3, 4), (2, 3, 4)],
        ["transpose12", "transpose12"],
        FP32,
        torch.device("cpu"),
        gen,
    )
    assert a.shape == b.shape == (2, 4, 3)
    assert torch.allclose(b, a + inputgen.CANCELLATION.delta)


def test_binary_matrix_contains_only_broadcastable_shapes():
    cfg = Config(
        dtypes=["fp32"], level="full", directions=["fwd"], include_large=False
    )
    for name in ("add", "sub", "mul"):
        spec = get_spec(name)
        cases = enumerate_cases([spec], cfg)
        assert cases
        for case in cases:
            shapes = spec.input_shapes(case.shape_case, case.params)
            views = spec.input_views(case.shape_case, case.params)
            effective = [
                inputgen.effective_shape(tuple(shape), view)
                for shape, view in zip(shapes, views)
            ]
            torch.broadcast_shapes(*effective)
            if case.pattern.kind is inputgen.PatternKind.CANCELLATION:
                assert effective[0] == effective[1]


def test_all_declared_grad_patterns_are_enumerated():
    spec = get_spec("add")
    cfg = Config(dtypes=["fp32"], level="smoke", directions=["bwd"])
    cases = enumerate_cases([spec], cfg)
    actual = {case.grad_pattern.label for case in cases if case.grad_pattern}
    expected = {pattern.label for pattern in spec.grad_patterns}
    assert actual == expected
    assert all("__grad-" in case.case_id for case in cases)


def test_smoke_matrix_includes_noncontiguous_shape():
    spec = get_spec("relu")
    cfg = Config(dtypes=["fp32"], level="smoke", directions=["fwd"])
    cases = enumerate_cases([spec], cfg)
    assert "noncontig" in {case.shape_case.category for case in cases}


def test_empty_backward_is_executed_and_validated():
    cfg = Config(secondary_check=False, dump_on_fail=False)
    good = execute_case(_case(), _identity_spec(), torch.device("cpu"), cfg)
    assert good.passed
    assert any(sub["label"] == "grad0" for sub in good.sub)

    bad_spec = _identity_spec(ppufn=lambda x: x.reshape(0, 1))
    bad = execute_case(_case(), bad_spec, torch.device("cpu"), cfg)
    assert not bad.passed


def test_backward_preserves_noncontiguous_input_stride():
    observed = {}

    def capture_stride(x):
        observed["stride"] = x.stride()
        return x * 1.0

    spec = _identity_spec(ppufn=capture_stride)
    case = _case(shape=(4, 8), view="slice2")
    cfg = Config(secondary_check=False, dump_on_fail=False)
    result = execute_case(case, spec, torch.device("cpu"), cfg)
    assert result.passed
    assert observed["stride"][-1] == 2


def test_relu6_reference_uses_hardtanh_boundary_gradient():
    spec = get_spec("relu6")
    x = torch.tensor([0.0, 6.0], dtype=torch.float64, requires_grad=True)
    spec.reffn(x).sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))


def test_repro_command_selects_exact_case_and_seed():
    case = _case("relu", shape=(4,))
    cmd = repro_command(case, Config(secondary_check=False))
    assert f"--case-id {case.case_id}" in cmd
    assert f"--case-seed {case.seed}" in cmd
    assert "--no-secondary" in cmd


def test_repro_script_uses_framework_root_for_custom_output(tmp_path):
    case = _case()
    cfg = Config(secondary_check=False, dump_on_fail=False)
    dump_dir = dump_case(
        case, _identity_spec(), torch.device("cpu"), cfg, str(tmp_path / "custom")
    )
    with open(os.path.join(dump_dir, "repro.sh")) as fh:
        script = fh.read()
    assert f"cd {shlex.quote(_ROOT)}" in script
    assert f"--case-seed {case.seed}" in script


def test_ci_rejects_missing_accelerator():
    ci_path = os.path.join(_ROOT, "..", ".ci", "ppu", "op_accuracy_test.sh")
    with open(ci_path) as fh:
        script = fh.read()
    assert "sys.exit(0 if available and count > 0 else 1)" in script
