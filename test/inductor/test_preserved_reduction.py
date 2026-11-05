# Owner(s): ["module: inductor"]
import ast
import dataclasses
import hashlib
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from torch._inductor.codegen.gluon import GluonCodegen
from torch._inductor.runtime.preserved_reduction import (
    BlockedReductionLayout,
    extract_reduction_encoding,
    PreservedReductionCall,
    ReductionCacheIdentity,
    ReductionLaunchConfig,
    UnsupportedReductionContract,
)
from torch._inductor.test_case import TestCase as InductorTestCase


# Inductor's TestCase isolates the compile/choice caches per test and supports the
# standard test driver's shard options, unlike plain unittest.TestCase.
class TestCase(InductorTestCase):
    pass


TTGIR = '''
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  %result = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<2x64xf32, #blocked>) -> tensor<2xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
}
'''

SIMD_SOURCE = '''
@triton.jit
def kernel(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    rbase = tl.arange(0, R0_BLOCK)[None, :]
    acc = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        rindex = r0_offset + rbase
        mask = (xindex < xnumel) & (rindex < r0_numel)
        value = tl.load(in_ptr0 + xindex * r0_numel + rindex, mask, other=0.0)
        product = value * value
        product = tl.broadcast_to(product, [XBLOCK, R0_BLOCK])
        acc = tl.where(mask, acc + product, acc)
        sigmoid = tl.sigmoid(value)
        tl.store(out_ptr1 + xindex * r0_numel + rindex, (sigmoid * value).to(tl.bfloat16), mask)
    result = tl.sum(acc, 1)[:, None]
    tl.store(out_ptr0 + xindex, result, xindex < xnumel)
'''


class PreservedReductionTests(TestCase):
    def config(self, **kwargs):
        values = dict(kwargs={"XBLOCK": 2, "R0_BLOCK": 64}, num_warps=2, num_stages=1, num_ctas=1)
        values.update(kwargs)
        return ReductionLaunchConfig.from_config(SimpleNamespace(**values))

    def test_encoding(self):
        encoding = extract_reduction_encoding(TTGIR)
        self.assertEqual(encoding.layout, BlockedReductionLayout((1, 1), (1, 32), (1, 2), (1, 0)))
        self.assertEqual(encoding.shape, (2, 64))
        encoding.validate_config(self.config())
        self.assertEqual(pickle.loads(pickle.dumps(encoding)), encoding)

    def test_encoding_rejects_unknown(self):
        invalid = [
            TTGIR.replace("axis = 1", "axis = 0"),
            TTGIR.replace("arith.addf", "arith.maximumf"),
            TTGIR.replace("%a, %b", "%a, %a"),
            TTGIR.replace("tt.reduce.return %sum", "tt.reduce.return %a"),
            TTGIR.replace("sizePerThread = [1, 1]", "sizePerThread = [1, 3]"),
            TTGIR.replace("order = [1, 0]", "order = [1, 1]"),
            TTGIR.replace("order = [1, 0]", "order = [1, 0], CTAsPerCGA = [1, 1]"),
            TTGIR.replace('"ttg.num-warps" = 2', '"ttg.num-warps" = 4'),
            TTGIR.replace('"ttg.num-ctas" = 1', '"ttg.num-ctas" = 2'),
            TTGIR + TTGIR,
            TTGIR.replace("#ttg.blocked", "#ttg.linear"),
        ]
        for source in invalid:
            with self.subTest(source=source):
                with self.assertRaises(UnsupportedReductionContract):
                    extract_reduction_encoding(source)
        with self.assertRaises(UnsupportedReductionContract):
            extract_reduction_encoding(TTGIR).validate_config(self.config(kwargs={"XBLOCK": 1, "R0_BLOCK": 64}))

    def test_config_rejects_unsupported(self):
        for fields in (
            dict(num_ctas=2), dict(pre_hook=lambda args: None),
            dict(kwargs={"XBLOCK": 2, "R0_BLOCK": 64, "R1_BLOCK": 64}),
            dict(kwargs={"XBLOCK": 2, "R0_BLOCK": 0}),
        ):
            with self.subTest(fields=fields):
                with self.assertRaises(UnsupportedReductionContract):
                    self.config(**fields)

    def test_cache_identity(self):
        identity = ReductionCacheIdentity.create(
            source=SIMD_SOURCE, config=self.config(),
            encoding=extract_reduction_encoding(TTGIR),
            target=SimpleNamespace(backend="cuda", arch=89, warp_size=32),
            compiler_version="3.7.0", backend_hash="ppu-version",
            compile_options={"enable_fp_fusion": True, "num_stages": 1},
        )
        self.assertEqual(identity.cache_key(), pickle.loads(pickle.dumps(identity)).cache_key())
        changes = (
            dict(source_hash="other-source"), dict(compiler_version="3.7.1"),
            dict(backend_hash="other-backend"), dict(target=("cuda", 90, 32)),
            dict(compile_options_json='{"enable_fp_fusion":false}'),
            dict(contract_version=identity.contract_version + 1),
            dict(encoding=dataclasses.replace(identity.encoding, layout=BlockedReductionLayout((1, 2), (1, 32), (1, 2), (1, 0)))),
            dict(config=dataclasses.replace(identity.config, num_stages=2)),
            dict(gluon_source_hash="other-gluon"),
            dict(candidate_arg_map=((0, 1),)),
        )
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(identity.cache_key(), dataclasses.replace(identity, **change).cache_key())

    def test_gluon_source(self):
        result = GluonCodegen.generate(SIMD_SOURCE)
        self.assertIn("SliceLayout(1, INDUCTOR_REDUCTION_LAYOUT)", result.source)
        self.assertIn("SliceLayout(0, INDUCTOR_REDUCTION_LAYOUT)", result.source)
        self.assertIn("gl.sum(acc, 1)", result.source)
        self.assertIn(".to(gl.bfloat16)", result.source)
        self.assertIn("acc + product", result.source)
        self.assertIn("1 / (1 + gl.exp(-value))", result.source)
        self.assertNotIn("tl.", result.source)
        self.assertEqual(result.argument_names[-2:], ("XBLOCK", "R0_BLOCK"))
        ast.parse(result.source)
        self.assertEqual(pickle.loads(pickle.dumps(result)), result)

    def test_gluon_rejects_unsupported(self):
        for source in (
            SIMD_SOURCE.replace("tl.sum(acc, 1)", "tl.max(acc, 1)"),
            SIMD_SOURCE.replace("tl.sum(acc, 1)", "tl.sum(acc, 0)"),
            SIMD_SOURCE.replace("tl.sigmoid(value)", "libdevice.tanh(value)"),
            SIMD_SOURCE.replace("tl.store(out_ptr0", "tl.atomic_add(out_ptr0"),
            SIMD_SOURCE.replace("range(0, r0_numel, R0_BLOCK)", "range(0, r0_numel, 32)"),
            SIMD_SOURCE.replace("tl.arange(0, XBLOCK)", "tl.arange(0, 16)"),
            SIMD_SOURCE.replace("value * value", "value ** 2"),
        ):
            with self.subTest(source=source):
                with self.assertRaises(UnsupportedReductionContract):
                    GluonCodegen.generate(source)

    def test_ast_source_selection(self):
        from torch._inductor.runtime import triton_heuristics

        normal = SimpleNamespace(is_gluon=lambda: False)
        gluon = SimpleNamespace(is_gluon=lambda: True)
        if triton_heuristics.ASTSource is not None:
            self.assertIs(triton_heuristics._get_ast_source(normal), triton_heuristics.ASTSource)
        else:
            # Without Triton, triton_compat sets ASTSource=None; normal dispatch must
            # raise instead of silently returning None.
            with self.assertRaises(RuntimeError):
                triton_heuristics._get_ast_source(normal)
        try:
            from triton.experimental.gluon._runtime import GluonASTSource
        except ImportError:
            with self.assertRaises(ImportError):
                triton_heuristics._get_ast_source(gluon)
        else:
            self.assertIs(triton_heuristics._get_ast_source(gluon), GluonASTSource)
        with patch.object(triton_heuristics, "ASTSource", None):
            with self.assertRaises(RuntimeError):
                triton_heuristics._get_ast_source(normal)

    @staticmethod
    def sequence_call():
        pointwise = SimpleNamespace(run=MagicMock())
        reduction = SimpleNamespace(run=MagicMock())
        call = PreservedReductionCall(
            reduction,
            pointwise_kernel=pointwise,
            gluon_source="source",
            gluon_function_name="kernel",
            gluon_arg_names=("input", "output", "xnumel", "XBLOCK", "R0_BLOCK"),
            gluon_layout_arg="LAYOUT",
            pointwise_arg_count=1,
            reduction_arg_count=1,
            candidate_arg_map=((0, 0), (1, 0)),
            candidate_output_indices=(1,),
            candidate_xnumel_index=2,
        )
        return call, pointwise, reduction

    def test_sequence_contract_fallback_keeps_both_standard_launches(self):
        call, pointwise, reduction = self.sequence_call()
        with patch.object(
            call,
            "_contract",
            side_effect=UnsupportedReductionContract("unsupported"),
        ):
            call.pointwise.run("pointwise", stream=1)
            call.reduction.run("reduction", stream=1)
        pointwise.run.assert_called_once_with("pointwise", stream=1)
        reduction.run.assert_called_once_with("reduction", stream=1)
        self.assertEqual(call._selected_mode, "standard")

        call.pointwise.run("next-pointwise", stream=2)
        call.reduction.run("next-reduction", stream=2)
        pointwise.run.assert_called_with("next-pointwise", stream=2)
        reduction.run.assert_called_with("next-reduction", stream=2)

    def test_sequence_selects_gluon_after_complete_pair_benchmark(self):
        call, pointwise, reduction = self.sequence_call()
        identity = MagicMock()
        with (
            patch.object(call, "_contract", return_value=identity),
            patch.object(call, "_load_choice", return_value=None),
            patch.object(call, "_load_candidate"),
            patch.object(call, "_validate_candidate", return_value=True),
            patch.object(call, "_benchmark", return_value=(2.0, 1.0)),
        ):
            call.pointwise.run("pointwise")
            call.reduction.run("reduction")
        pointwise.run.assert_called_once_with("pointwise")
        reduction.run.assert_called_once_with("reduction")
        self.assertEqual(call._selected_mode, "gluon")

        pointwise.run.reset_mock()
        reduction.run.reset_mock()
        with patch.object(call, "_run_candidate", return_value="candidate") as run:
            call.pointwise.run("next-pointwise")
            self.assertEqual(call.reduction.run("next-reduction"), "candidate")
        run.assert_called_once()
        pointwise.run.assert_not_called()
        reduction.run.assert_not_called()

    def test_cached_gluon_is_revalidated_without_rebenchmark(self):
        call, _, _ = self.sequence_call()
        with (
            patch.object(call, "_contract", return_value=MagicMock()),
            patch.object(call, "_load_choice", return_value="gluon"),
            patch.object(call, "_load_candidate"),
            patch.object(call, "_validate_candidate", return_value=True),
            patch.object(call, "_benchmark") as benchmark,
        ):
            call.pointwise.run("pointwise")
            call.reduction.run("reduction")
        benchmark.assert_not_called()
        self.assertEqual(call._selected_mode, "gluon")

    def test_reduction_endpoint_requires_matching_pointwise(self):
        call, _, _ = self.sequence_call()
        with self.assertRaisesRegex(RuntimeError, "no matching pointwise call"):
            call.reduction.run("reduction")

    @staticmethod
    def fast_path_call():
        """Steady-state selector with precomputed launch arguments; argument mapping includes xnumel."""
        pointwise = SimpleNamespace(run=MagicMock())
        reduction = SimpleNamespace(run=MagicMock())
        call = PreservedReductionCall(
            reduction,
            pointwise_kernel=pointwise,
            gluon_source="source",
            gluon_function_name="kernel",
            gluon_arg_names=("input", "output", "xnumel", "XBLOCK", "R0_BLOCK"),
            gluon_layout_arg="LAYOUT",
            pointwise_arg_count=2,
            reduction_arg_count=2,
            candidate_arg_map=((0, 0), (1, 0), (0, 1)),
            candidate_output_indices=(1,),
            candidate_xnumel_index=2,
        )
        call._selected_mode = "gluon"
        call._candidate = MagicMock()
        call._launch_kwargs = {
            "XBLOCK": 2,
            "R0_BLOCK": 64,
            "LAYOUT": "layout",
            "num_warps": 2,
            "num_stages": 1,
            "num_ctas": 1,
        }
        return call, pointwise, reduction

    def test_fast_path_maps_arguments_and_caches_grid(self):
        call, pointwise, reduction = self.fast_path_call()
        call.pointwise.run("input", 17)
        result = call.reduction.run("output", "extra")
        call._candidate.__getitem__.assert_called_once_with((9, 1, 1))
        launch = call._candidate.__getitem__.return_value
        self.assertIs(result, launch.return_value)
        launch.assert_called_once_with(
            "input", "output", 17,
            XBLOCK=2, R0_BLOCK=64, LAYOUT="layout",
            num_warps=2, num_stages=1, num_ctas=1,
        )
        pointwise.run.assert_not_called()
        reduction.run.assert_not_called()
        # Same xnumel reuses the grid; a change triggers recomputation.
        call.pointwise.run("input2", 17)
        call.reduction.run("output2", "extra")
        self.assertEqual(call._candidate.__getitem__.call_count, 2)
        call.pointwise.run("input3", 33)
        call.reduction.run("output3", "extra")
        call._candidate.__getitem__.assert_called_with((17, 1, 1))

    def test_fast_path_non_int_xnumel_falls_back_to_full_path(self):
        call, _, _ = self.fast_path_call()
        with patch.object(call, "_run_candidate", return_value="full") as full:
            call.pointwise.run("input", "symbolic")
            self.assertEqual(call.reduction.run("output", "extra"), "full")
        full.assert_called_once()
        call._candidate.__getitem__.assert_not_called()

    def test_fast_path_argument_count_change_raises(self):
        call, _, _ = self.fast_path_call()
        call.pointwise.run("input")
        with self.assertRaisesRegex(UnsupportedReductionContract, "argument count changed"):
            call.reduction.run("output", "extra")

    def test_fast_path_without_precomputed_kwargs_uses_full_path(self):
        call, _, _ = self.fast_path_call()
        call._launch_kwargs = None
        with patch.object(call, "_run_candidate", return_value="full") as full:
            call.pointwise.run("input", 17)
            self.assertEqual(call.reduction.run("output", "extra"), "full")
        full.assert_called_once()

    def test_fast_path_grid_cache_is_thread_local(self):
        import threading

        call, _, _ = self.fast_path_call()
        seen: dict[int, list] = {}
        barrier = threading.Barrier(2)

        def getitem(grid):
            # Record the grid actually used for launch, per current thread.
            seen.setdefault(threading.get_ident(), []).append(grid)
            return MagicMock()

        call._candidate.__getitem__.side_effect = getitem

        def worker(xnumel: int) -> None:
            call.pointwise.run("input", xnumel)
            barrier.wait()
            call.reduction.run("output", "extra")

        threads = [threading.Thread(target=worker, args=(n,)) for n in (17, 33)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Each thread only sees its own grid, never overwritten by another thread's xnumel.
        self.assertEqual(sorted(seen.values()), [[(9, 1, 1)], [(17, 1, 1)]])

    def test_bitwise_comparison_distinguishes_signed_zero_and_accepts_nan_bits(self):
        positive_zero = torch.tensor([0.0], dtype=torch.float32)
        negative_zero = torch.tensor([-0.0], dtype=torch.float32)
        self.assertFalse(
            PreservedReductionCall._bitwise_equal(positive_zero, negative_zero)
        )
        nan_bits = torch.tensor([0x7FC00001], dtype=torch.int32)
        left = nan_bits.view(torch.float32)
        right = nan_bits.clone().view(torch.float32)
        self.assertTrue(PreservedReductionCall._bitwise_equal(left, right))

    def test_fp_fusion_option_reads_standard_meta(self):
        # #4: the candidate inherits the standard kernel's FMA contraction option.
        for value in (True, False):
            kernel = SimpleNamespace(triton_meta={"enable_fp_fusion": value})
            self.assertIs(PreservedReductionCall._fp_fusion_option(kernel), value)

    def test_fp_fusion_option_rejects_missing_or_non_bool(self):
        for meta in (None, {}, {"enable_fp_fusion": 1}, {"enable_fp_fusion": "true"}):
            with self.subTest(meta=meta):
                with self.assertRaisesRegex(UnsupportedReductionContract, "enable_fp_fusion"):
                    PreservedReductionCall._fp_fusion_option(SimpleNamespace(triton_meta=meta))

    def _contract_call(self, *, standard_fp_fusion, pointwise_fp_fusion):
        call, pointwise, reduction = self.sequence_call()
        reduction.triton_meta = {"enable_fp_fusion": standard_fp_fusion}
        pointwise.triton_meta = {"enable_fp_fusion": pointwise_fp_fusion}
        reduction.device_props = SimpleNamespace(type="cuda", cc=89, warp_size=32)
        reduction.fn = SimpleNamespace(src=SIMD_SOURCE)
        launcher = SimpleNamespace(config=SimpleNamespace(
            kwargs={"XBLOCK": 2, "R0_BLOCK": 64}, num_warps=2, num_stages=1, num_ctas=1,
        ))
        result = MagicMock()
        result.compile_meta = {
            "enable_fp_fusion": standard_fp_fusion, "num_warps": 2, "num_stages": 1,
        }
        result.kernel.hash = "kernel-hash"
        return call, pointwise, reduction, launcher, result

    def test_contract_rejects_fp_fusion_disagreement(self):
        # #4: pointwise and reduction compiled with different FMA settings cannot share a candidate.
        call, pointwise, reduction, launcher, result = self._contract_call(
            standard_fp_fusion=True, pointwise_fp_fusion=False,
        )
        with (
            patch.object(call, "_selected_compile_result", return_value=(launcher, result)),
            patch.object(call, "_read_ttgir", return_value=TTGIR),
        ):
            with self.assertRaisesRegex(UnsupportedReductionContract, "disagree on enable_fp_fusion"):
                call._contract(reduction)

    def test_contract_inherits_fp_fusion_and_gluon_identity(self):
        # #4 + #7: fp_fusion flows into the launch kwargs; Gluon source and arg map enter the identity.
        call, pointwise, reduction, launcher, result = self._contract_call(
            standard_fp_fusion=False, pointwise_fp_fusion=False,
        )
        with (
            patch.object(call, "_selected_compile_result", return_value=(launcher, result)),
            patch.object(call, "_read_ttgir", return_value=TTGIR),
            patch.object(BlockedReductionLayout, "to_gluon", return_value="layout"),
        ):
            identity = call._contract(reduction)
        self.assertIs(call._candidate_fp_fusion, False)
        self.assertIs(call._launch_kwargs["enable_fp_fusion"], False)
        self.assertEqual(identity.gluon_source_hash, hashlib.sha256(b"source").hexdigest())
        self.assertEqual(identity.candidate_arg_map, ((0, 0), (1, 0)))

    def test_run_candidate_passes_fp_fusion(self):
        # #4: the full candidate launch forwards the inherited enable_fp_fusion option.
        call, _, _ = self.fast_path_call()
        call._launch_kwargs = None  # force the full validation path
        call._candidate_config = self.config()
        call._candidate_layout = "layout"
        call._candidate_fp_fusion = False
        call.pointwise.run("input", 17)
        call.reduction.run("output", "extra")
        launch = call._candidate.__getitem__.return_value
        launch.assert_called_once_with(
            "input", "output", 17,
            XBLOCK=2, R0_BLOCK=64, LAYOUT="layout",
            num_warps=2, num_stages=1, num_ctas=1, enable_fp_fusion=False,
        )

    def test_compile_errors_fall_back_to_standard(self):
        # #3: Triton compilation/resource errors are not OSError; they must still fall back.
        try:
            from triton.compiler.errors import CompilationError
            from triton.runtime.errors import OutOfResources, PTXASError
        except ImportError:
            self.skipTest("Triton errors unavailable")
        errors = (
            CompilationError(None, None, "boom"),
            OutOfResources(128, 64, "shared memory"),
            PTXASError("boom"),
        )
        for stage in ("_validate_candidate", "_benchmark"):
            for exc in errors:
                with self.subTest(stage=stage, exc=type(exc).__name__):
                    call, pointwise, reduction = self.sequence_call()
                    overrides = {
                        "_contract": MagicMock(return_value=MagicMock()),
                        "_load_choice": MagicMock(return_value=None),
                        "_load_candidate": MagicMock(),
                        "_validate_candidate": MagicMock(return_value=True),
                        "_benchmark": MagicMock(return_value=(2.0, 1.0)),
                    }
                    overrides[stage] = MagicMock(side_effect=exc)
                    with patch.multiple(call, **overrides):
                        call.pointwise.run("pointwise")
                        call.reduction.run("reduction")
                    self.assertEqual(call._selected_mode, "standard")
                    pointwise.run.assert_called_once_with("pointwise")
                    reduction.run.assert_called_once_with("reduction")

    def test_standard_path_consumes_pointwise_stashed_before_selection(self):
        # #2: a pointwise call deferred while no mode was selected must still run once the
        # standard mode becomes visible (e.g. selected by another thread).
        call, pointwise, reduction = self.sequence_call()
        call.pointwise.run("pointwise", stream=1)
        call._selected_mode = "standard"  # another thread completed the one-time selection
        call.reduction.run("reduction", stream=1)
        pointwise.run.assert_called_once_with("pointwise", stream=1)
        reduction.run.assert_called_once_with("reduction", stream=1)

    def test_concurrent_selection_serializes_and_consumes_pending(self):
        # #2: the selection lock serializes the one-time choice; every thread's deferred
        # pointwise still runs exactly once whether it selected or observed the mode.
        import threading

        call, pointwise, reduction = self.sequence_call()
        barrier = threading.Barrier(2)

        def worker(tag):
            call.pointwise.run(f"pw-{tag}", stream=tag)
            barrier.wait()
            call.reduction.run(f"red-{tag}", stream=tag)

        with patch.object(call, "_contract", side_effect=UnsupportedReductionContract("unsupported")):
            threads = [threading.Thread(target=worker, args=(t,)) for t in (1, 2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(call._selected_mode, "standard")
        self.assertEqual(pointwise.run.call_count, 2)
        self.assertEqual(reduction.run.call_count, 2)
        pointwise.run.assert_any_call("pw-1", stream=1)
        pointwise.run.assert_any_call("pw-2", stream=2)
        reduction.run.assert_any_call("red-1", stream=1)
        reduction.run.assert_any_call("red-2", stream=2)

    def test_cache_identity_separates_gluon_source_and_arg_map(self):
        # #7: different Gluon payloads or argument mappings must not share a timing decision.
        base = dict(
            source=SIMD_SOURCE, config=self.config(),
            encoding=extract_reduction_encoding(TTGIR),
            target=SimpleNamespace(backend="cuda", arch=89, warp_size=32),
            compiler_version="3.7.0", backend_hash="ppu-version",
            compile_options={"enable_fp_fusion": True, "num_stages": 1},
        )
        identity = ReductionCacheIdentity.create(
            gluon_source="gluon-a", candidate_arg_map=((0, 0), (1, 0)), **base
        )
        self.assertEqual(identity.gluon_source_hash, hashlib.sha256(b"gluon-a").hexdigest())
        self.assertEqual(identity.candidate_arg_map, ((0, 0), (1, 0)))
        other_source = ReductionCacheIdentity.create(
            gluon_source="gluon-b", candidate_arg_map=((0, 0), (1, 0)), **base
        )
        self.assertNotEqual(identity.cache_key(), other_source.cache_key())
        other_map = ReductionCacheIdentity.create(
            gluon_source="gluon-a", candidate_arg_map=((0, 0), (1, 1)), **base
        )
        self.assertNotEqual(identity.cache_key(), other_map.cache_key())

    def test_contract_compiler_version_unknown_without_triton(self):
        # Without Triton installed, `import triton` raises ImportError (not just
        # AttributeError); the identity must still be constructible on CPU-only hosts.
        call, pointwise, reduction, launcher, result = self._contract_call(
            standard_fp_fusion=True, pointwise_fp_fusion=True,
        )
        with (
            patch.object(call, "_selected_compile_result", return_value=(launcher, result)),
            patch.object(call, "_read_ttgir", return_value=TTGIR),
            patch.object(BlockedReductionLayout, "to_gluon", return_value="layout"),
            patch.dict(sys.modules, {"triton": None}),
        ):
            identity = call._contract(reduction)
        self.assertEqual(identity.compiler_version, "unknown")

    def test_is_capturing_checks_device_stream(self):
        call, _, _ = self.sequence_call()
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertTrue(call._is_capturing())
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            self.assertFalse(call._is_capturing())
        call._device_str = "unsupported_device"
        self.assertFalse(call._is_capturing())

    def test_capture_runs_standard_and_defers_selection(self):
        # Validation/benchmarking synchronize with the host, which is forbidden during
        # graph capture; the captured call must run the standard sequence without
        # selecting, and a later non-capturing call must still perform the choice.
        call, pointwise, reduction = self.sequence_call()
        with (
            patch.object(call, "_is_capturing", return_value=True),
            patch.object(call, "_contract") as contract,
        ):
            call.pointwise.run("captured-pointwise")
            call.reduction.run("captured-reduction")
            contract.assert_not_called()
        self.assertIsNone(call._selected_mode)
        pointwise.run.assert_called_once_with("captured-pointwise")
        reduction.run.assert_called_once_with("captured-reduction")
        with (
            patch.object(call, "_is_capturing", return_value=False),
            patch.object(call, "_contract", side_effect=UnsupportedReductionContract("unsupported")),
        ):
            call.pointwise.run("later-pointwise")
            call.reduction.run("later-reduction")
        self.assertEqual(call._selected_mode, "standard")
        self.assertEqual(pointwise.run.call_count, 2)
        self.assertEqual(reduction.run.call_count, 2)

    def test_validation_oom_falls_back_to_standard(self):
        # Candidate validation allocates full output copies; an OOM there must reject
        # the optional optimization instead of failing the model invocation.
        for stage in ("_validate_candidate", "_benchmark"):
            with self.subTest(stage=stage):
                call, pointwise, reduction = self.sequence_call()
                overrides = {
                    "_contract": MagicMock(return_value=MagicMock()),
                    "_load_choice": MagicMock(return_value=None),
                    "_load_candidate": MagicMock(),
                    "_validate_candidate": MagicMock(return_value=True),
                    "_benchmark": MagicMock(return_value=(2.0, 1.0)),
                }
                overrides[stage] = MagicMock(
                    side_effect=torch.OutOfMemoryError("out of memory")
                )
                with patch.multiple(call, **overrides):
                    call.pointwise.run("pointwise")
                    call.reduction.run("reduction")
                self.assertEqual(call._selected_mode, "standard")
                pointwise.run.assert_called_once_with("pointwise")
                reduction.run.assert_called_once_with("reduction")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests(needs="filelock")
