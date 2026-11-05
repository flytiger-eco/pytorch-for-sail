# Owner(s): ["module: inductor"]
import ast
import dataclasses
import pickle
import unittest
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


class PreservedReductionTests(unittest.TestCase):
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
        self.assertIs(triton_heuristics._get_ast_source(normal), triton_heuristics.ASTSource)
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


if __name__ == "__main__":
    unittest.main()
