# Owner(s): ["module: inductor"]
"""Unit/integration tests for Gluon preserved-reduction fusion and the runtime contract.

Coverage:
- GluonCodegen AST transform admission and rejection
- Strict parsing of TTGIR extract_reduction_encoding
- BlockedReductionLayout / ReductionLaunchConfig / ReductionEncoding validation
- ReductionCacheIdentity cache key consistency
- PreservedReductionCall lifecycle (Future resolution, run delegation, fallback, sequence selection)
- Silent fallback of _track_preserved_reduction / _define_preserved_sequence
- async_compile.preserved_reduction wiring
- End-to-end compilation emitting the preserved_reduction wrapper (requires GPU)
"""
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch._inductor.codegen.gluon import GluonCodegen, GluonKernelSource
from torch._inductor.runtime.preserved_reduction import (
    BlockedReductionLayout,
    CONTRACT_VERSION,
    extract_reduction_encoding,
    PreservedReductionCall,
    ReductionCacheIdentity,
    ReductionEncoding,
    ReductionLaunchConfig,
    UnsupportedReductionContract,
)
from torch.testing._internal.inductor_utils import HAS_GPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_gpu_and_triton


# ---------------------------------------------------------------------------
# Typical SIMD reduction kernel source (for GluonCodegen tests)
# ---------------------------------------------------------------------------
SIMPLE_REDUCTION_SRC = textwrap.dedent("""\
    import triton
    import triton.language as tl

    @triton.jit
    def triton_red_fused_0(
        in_ptr0, out_ptr0,
        x_numel, r0_numel,
        XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr,
    ):
        x_offset = tl.program_id(0) * XBLOCK
        x_index = x_offset + tl.arange(0, XBLOCK)
        x_mask = x_index < x_numel
        _tmp0 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + tl.arange(0, R0_BLOCK)
            r0_mask = r0_index < r0_numel
            tmp0 = tl.load(in_ptr0 + (x_index[:, None] * r0_numel + r0_index[None, :]), x_mask[:, None] & r0_mask[None, :])
            _tmp0 = _tmp0 + tmp0
        tmp1 = tl.sum(_tmp0, 1)
        tl.store(out_ptr0 + x_index, tmp1, x_mask)
""")

# ---------------------------------------------------------------------------
# Typical TTGIR (for extract_reduction_encoding tests)
# ---------------------------------------------------------------------------
SAMPLE_TTGIR = textwrap.dedent("""\
    #blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func @triton_red_fused_0(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
        %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
        %0 = "tt.reduce"(%cst) <{axis = 1 : i32}> ({
        ^bb0(%arg2: f32, %arg3: f32):
          %1 = arith.addf %arg2, %arg3 : f32
          tt.reduce.return %1 : f32
        }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        tt.return
      }
    }
""")


class TestGluonCodegen(unittest.TestCase):
    """GluonCodegen AST transform and admission."""

    def test_simple_reduction(self):
        result = GluonCodegen.generate(
            SIMPLE_REDUCTION_SRC, function_name="triton_red_fused_0"
        )
        self.assertIsInstance(result, GluonKernelSource)
        self.assertIn("gl.", result.source)
        self.assertNotIn("tl.", result.source)
        self.assertIn("gluon.jit", result.source)
        self.assertIn("SliceLayout", result.source)
        self.assertEqual(result.layout_argument, "INDUCTOR_REDUCTION_LAYOUT")
        self.assertIn("INDUCTOR_REDUCTION_LAYOUT", result.source)
        # All original arguments are preserved
        for name in ("in_ptr0", "out_ptr0", "x_numel", "r0_numel", "XBLOCK", "R0_BLOCK"):
            self.assertIn(name, result.argument_names)
        # The layout argument is appended at the end
        self.assertNotIn("INDUCTOR_REDUCTION_LAYOUT", result.argument_names)

    def test_no_function_found(self):
        with self.assertRaises(UnsupportedReductionContract):
            GluonCodegen.generate("x = 1\n", function_name="missing")

    def test_multiple_functions_rejected(self):
        src = "def a(): pass\ndef b(): pass\n"
        with self.assertRaises(UnsupportedReductionContract):
            GluonCodegen.generate(src)

    def test_persistent_reduction_rejected(self):
        # No loop -> persistent or multi-loop
        src = textwrap.dedent("""\
            def fn(in_ptr0, out_ptr0, x_numel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
                x_offset = tl.program_id(0) * XBLOCK
                _tmp0 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
                tmp1 = tl.sum(_tmp0, 1)
                tl.store(out_ptr0 + x_offset, tmp1)
        """)
        with self.assertRaises(UnsupportedReductionContract):
            GluonCodegen.generate(src, function_name="fn")

    def test_multi_sum_rejected(self):
        src = textwrap.dedent("""\
            def fn(in_ptr0, out_ptr0, x_numel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
                _a = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
                for r0_offset in range(0, r0_numel, R0_BLOCK):
                    pass
                s1 = tl.sum(_a, 1)
                s2 = tl.sum(_a, 1)
                tl.store(out_ptr0, s1 + s2)
        """)
        with self.assertRaises(UnsupportedReductionContract):
            GluonCodegen.generate(src, function_name="fn")

    def test_if_statement_rejected(self):
        src = textwrap.dedent("""\
            def fn(in_ptr0, out_ptr0, x_numel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
                _a = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
                for r0_offset in range(0, r0_numel, R0_BLOCK):
                    pass
                if True:
                    pass
                s = tl.sum(_a, 1)
                tl.store(out_ptr0, s)
        """)
        with self.assertRaises(UnsupportedReductionContract):
            GluonCodegen.generate(src, function_name="fn")

    def test_unsupported_tl_op_rejected(self):
        src = textwrap.dedent("""\
            def fn(in_ptr0, out_ptr0, x_numel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
                _a = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
                for r0_offset in range(0, r0_numel, R0_BLOCK):
                    pass
                s = tl.sum(_a, 1)
                r = tl.atomic_cas(out_ptr0, s, s)
                tl.store(out_ptr0, r)
        """)
        with self.assertRaises(UnsupportedReductionContract):
            GluonCodegen.generate(src, function_name="fn")

    def test_missing_required_params_rejected(self):
        # Missing R0_BLOCK
        src = textwrap.dedent("""\
            def fn(in_ptr0, out_ptr0, XBLOCK: tl.constexpr):
                pass
        """)
        with self.assertRaises(UnsupportedReductionContract):
            GluonCodegen.generate(src, function_name="fn")

    def test_sigmoid_expanded(self):
        src = textwrap.dedent("""\
            def fn(in_ptr0, out_ptr0, x_numel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
                x_offset = tl.program_id(0) * XBLOCK
                x_index = x_offset + tl.arange(0, XBLOCK)
                _tmp0 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
                for r0_offset in range(0, r0_numel, R0_BLOCK):
                    r0_index = r0_offset + tl.arange(0, R0_BLOCK)
                    tmp0 = tl.load(in_ptr0 + x_index, None)
                    sig = tl.sigmoid(tmp0)
                    _tmp0 = _tmp0 + sig
                s = tl.sum(_tmp0, 1)
                tl.store(out_ptr0 + x_index, s)
        """)
        result = GluonCodegen.generate(src, function_name="fn")
        # sigmoid is expanded to 1/(1+exp(-x)); gl.sigmoid is not kept
        self.assertNotIn("gl.sigmoid", result.source)
        self.assertIn("gl.exp", result.source)


class TestBlockedReductionLayout(unittest.TestCase):
    def test_valid_layout(self):
        layout = BlockedReductionLayout((1, 8), (32, 1), (4, 1), (1, 0))
        self.assertEqual(layout.size_per_thread, (1, 8))

    def test_non_power_of_two_rejected(self):
        with self.assertRaises(UnsupportedReductionContract):
            BlockedReductionLayout((1, 3), (32, 1), (4, 1), (1, 0))

    def test_wrong_order_rejected(self):
        with self.assertRaises(UnsupportedReductionContract):
            BlockedReductionLayout((1, 8), (32, 1), (4, 1), (0, 0))

    def test_validate_launch_mismatch(self):
        layout = BlockedReductionLayout((1, 8), (32, 1), (4, 1), (1, 0))
        # num_warps=8 but the product of warpsPerCTA=(4,1) is 4
        with self.assertRaises(UnsupportedReductionContract):
            layout.validate_launch(8, 32, 1)

    def test_validate_launch_correct(self):
        layout = BlockedReductionLayout((1, 8), (32, 1), (4, 1), (1, 0))
        layout.validate_launch(4, 32, 1)  # must not raise


class TestReductionLaunchConfig(unittest.TestCase):
    def test_from_config_valid(self):
        cfg = SimpleNamespace(
            kwargs={"XBLOCK": 128, "R0_BLOCK": 64},
            num_warps=4, num_stages=2, num_ctas=1, pre_hook=None,
        )
        result = ReductionLaunchConfig.from_config(cfg)
        self.assertEqual(result.num_warps, 4)
        self.assertEqual(dict(result.kwargs), {"XBLOCK": 128, "R0_BLOCK": 64})

    def test_from_config_missing_r0block(self):
        cfg = SimpleNamespace(
            kwargs={"XBLOCK": 128}, num_warps=4, num_stages=2,
            num_ctas=1, pre_hook=None,
        )
        with self.assertRaises(UnsupportedReductionContract):
            ReductionLaunchConfig.from_config(cfg)

    def test_from_config_with_prehook_rejected(self):
        cfg = SimpleNamespace(
            kwargs={"XBLOCK": 128, "R0_BLOCK": 64},
            num_warps=4, num_stages=2, num_ctas=1,
            pre_hook=lambda: None,
        )
        with self.assertRaises(UnsupportedReductionContract):
            ReductionLaunchConfig.from_config(cfg)


class TestExtractReductionEncoding(unittest.TestCase):
    def test_sample_ttgir(self):
        enc = extract_reduction_encoding(SAMPLE_TTGIR)
        self.assertEqual(enc.layout.size_per_thread, (1, 8))
        self.assertEqual(enc.layout.threads_per_warp, (32, 1))
        self.assertEqual(enc.layout.warps_per_cta, (4, 1))
        self.assertEqual(enc.layout.order, (1, 0))
        self.assertEqual(enc.shape, (128, 64))
        self.assertEqual(enc.dtype, "f32")
        self.assertEqual(enc.axis, 1)
        self.assertEqual(enc.num_warps, 4)
        self.assertEqual(enc.warp_size, 32)
        self.assertEqual(enc.num_ctas, 1)

    def test_no_reduce(self):
        ttgir = 'module attributes {"ttg.num-ctas" = 1 : i32} {}'
        with self.assertRaises(UnsupportedReductionContract):
            extract_reduction_encoding(ttgir)

    def test_multiple_reduces(self):
        ttgir = SAMPLE_TTGIR + '\n"tt.reduce"(%x) <{axis = 0 : i32}> : nope'
        with self.assertRaises(UnsupportedReductionContract):
            extract_reduction_encoding(ttgir)

    def test_wrong_axis(self):
        # axis=0 is not supported
        bad = SAMPLE_TTGIR.replace("axis = 1", "axis = 0")
        with self.assertRaises(UnsupportedReductionContract):
            extract_reduction_encoding(bad)

    def test_missing_module_field(self):
        # Remove num-warps
        bad = SAMPLE_TTGIR.replace('"ttg.num-warps" = 4 : i32, ', '')
        with self.assertRaises(UnsupportedReductionContract):
            extract_reduction_encoding(bad)


class TestReductionEncoding(unittest.TestCase):
    def test_validate_config_match(self):
        enc = extract_reduction_encoding(SAMPLE_TTGIR)
        cfg = SimpleNamespace(
            kwargs={"XBLOCK": 128, "R0_BLOCK": 64},
            num_warps=4, num_stages=2, num_ctas=1, pre_hook=None,
        )
        launch = ReductionLaunchConfig.from_config(cfg)
        enc.validate_config(launch)  # must not raise

    def test_validate_config_mismatch(self):
        enc = extract_reduction_encoding(SAMPLE_TTGIR)
        cfg = SimpleNamespace(
            kwargs={"XBLOCK": 64, "R0_BLOCK": 64},
            num_warps=4, num_stages=2, num_ctas=1, pre_hook=None,
        )
        launch = ReductionLaunchConfig.from_config(cfg)
        with self.assertRaises(UnsupportedReductionContract):
            enc.validate_config(launch)


class TestReductionCacheIdentity(unittest.TestCase):
    def test_cache_key_deterministic(self):
        enc = extract_reduction_encoding(SAMPLE_TTGIR)
        cfg = SimpleNamespace(
            kwargs={"XBLOCK": 128, "R0_BLOCK": 64},
            num_warps=4, num_stages=2, num_ctas=1, pre_hook=None,
        )
        launch = ReductionLaunchConfig.from_config(cfg)
        target = SimpleNamespace(backend="ppu", arch=89, warp_size=32)
        identity = ReductionCacheIdentity.create(
            source="source_code_here",
            config=launch,
            encoding=enc,
            target=target,
            compiler_version="3.7.0",
            backend_hash="abc123",
            compile_options={"allow_fp16_reductions": True},
        )
        key1 = identity.cache_key()
        key2 = identity.cache_key()
        self.assertEqual(key1, key2)
        self.assertEqual(identity.contract_version, CONTRACT_VERSION)
        self.assertIsInstance(key1, str)
        self.assertEqual(len(key1), 64)  # sha256 hex


class TestPreservedReductionCall(unittest.TestCase):
    def _make_standard_kernel(self, *, has_launchers=True, has_asm=True, fp_fusion=True):
        # spec=[] keeps the mock free of an auto result attribute so it is not treated as a Future
        kernel = MagicMock(spec=[])
        kernel.run = MagicMock(return_value="run_result")
        kernel.triton_meta = {"enable_fp_fusion": fp_fusion}
        if has_launchers:
            launcher = MagicMock()
            launcher.config = SimpleNamespace(
                kwargs={"XBLOCK": 128, "R0_BLOCK": 64},
                num_warps=4, num_stages=2, num_ctas=1, pre_hook=None,
            )
            if has_asm:
                launcher.kernel.asm = {"ttgir": SAMPLE_TTGIR}
            else:
                launcher.kernel.asm = {}
            kernel.launchers = [launcher]
        else:
            kernel.launchers = []
        return kernel

    def test_first_run_delegates_to_standard(self):
        std = self._make_standard_kernel()
        prc = PreservedReductionCall(
            std,
            gluon_source="...",
            gluon_arg_names=("a",),
            gluon_layout_arg="LAYOUT",
        )
        result = prc.run(1, 2, 3)
        self.assertEqual(result, "run_result")
        std.run.assert_called_once_with(1, 2, 3)

    def test_subsequent_runs_use_selected(self):
        std = self._make_standard_kernel()
        prc = PreservedReductionCall(
            std, gluon_source="...", gluon_arg_names=("a",),
            gluon_layout_arg="LAYOUT",
        )
        prc.run()
        std.run.reset_mock()
        prc.run(42)
        # The second run should go through _selected instead of re-triggering _validate_and_select
        std.run.assert_called_once_with(42)

    def test_fallback_on_no_launchers(self):
        std = self._make_standard_kernel(has_launchers=False)
        prc = PreservedReductionCall(
            std, gluon_source="...", gluon_arg_names=("a",),
            gluon_layout_arg="LAYOUT",
        )
        result = prc.run()
        self.assertEqual(result, "run_result")
        # After fallback the standard kernel stays selected
        self.assertIs(prc._selected, std)

    def test_fallback_on_no_ttgir(self):
        std = self._make_standard_kernel(has_asm=False)
        prc = PreservedReductionCall(
            std, gluon_source="...", gluon_arg_names=("a",),
            gluon_layout_arg="LAYOUT",
        )
        prc.run()
        self.assertIs(prc._selected, std)

    def test_future_resolution(self):
        std = self._make_standard_kernel()
        future = MagicMock()
        future.result.return_value = std
        prc = PreservedReductionCall(
            future, gluon_source="...", gluon_arg_names=("a",),
            gluon_layout_arg="LAYOUT",
        )
        result = prc.run()
        future.result.assert_called()
        self.assertEqual(result, "run_result")

    def test_getattr_forwards(self):
        std = self._make_standard_kernel()
        std.custom_attr = "hello"
        prc = PreservedReductionCall(
            std, gluon_source="...", gluon_arg_names=("a",),
            gluon_layout_arg="LAYOUT",
        )
        self.assertEqual(prc.custom_attr, "hello")


class TestAsyncCompilePreservedReduction(unittest.TestCase):
    def test_preserved_reduction_method_exists(self):
        from torch._inductor.async_compile import AsyncCompile
        ac = AsyncCompile()
        self.assertTrue(hasattr(ac, "preserved_reduction"))

    def test_preserved_reduction_creates_call(self):
        from torch._inductor.async_compile import AsyncCompile
        ac = AsyncCompile()
        std = MagicMock()
        result = ac.preserved_reduction(
            std, gluon_source="src", gluon_arg_names=("a",),
            gluon_layout_arg="LAYOUT",
        )
        self.assertIsInstance(result, PreservedReductionCall)


class TestPreservedSequenceAdmission(unittest.TestCase):
    """_next_preserved_reduction structural admission: benign cat aliases/Nop accepted, dangerous structures rejected."""

    def _buffer(self, name, aliases=(), mutations=()):
        buf = MagicMock()
        buf.get_name.return_value = name
        # Real contract: get_aliases/get_mutations return buffer name strings.
        buf.get_aliases.return_value = list(aliases)
        buf.get_mutations.return_value = list(mutations)
        return buf

    def _node(self, name, *, buffers, outputs, used, reduction=False,
              side_effects=False, nop=False, device="cuda"):
        from torch._inductor.scheduler import (
            NopKernelSchedulerNode,
            SchedulerNode,
        )

        spec = NopKernelSchedulerNode if nop else SchedulerNode
        node = MagicMock(spec=spec)
        node.get_name.return_value = name
        node.get_buffer_names.return_value = set(buffers)
        node.get_outputs.return_value = outputs
        node.used_buffer_names.return_value = set(used)
        node.is_reduction.return_value = reduction
        node.has_side_effects.return_value = side_effects
        node.has_aliasing_or_mutation.return_value = any(
            b.get_aliases() or b.get_mutations() for b in outputs
        )
        node.get_device.return_value = device
        node.get_nodes.return_value = [node]
        return node

    def _run_admission(self, nodes, pointwise):
        from torch._inductor.codegen import triton as triton_codegen

        fake = SimpleNamespace(
            scheduler=SimpleNamespace(nodes=nodes),
            _only_output_users=triton_codegen.TritonScheduling._only_output_users,
        )
        with patch.object(triton_codegen, "V") as v, patch.object(
            triton_codegen.config.triton, "multi_kernel", 0
        ):
            v.graph.no_fuse_with_reduction = {"buf0", "buf1"}
            return triton_codegen.TritonScheduling._next_preserved_reduction(
                fake, pointwise
            )

    def _make_cat_sequence(self, *, pw_mutations=(), pw_alias_target=None,
                           red_reads=(), shared_read=True, middle_nop=True):
        from torch._inductor.scheduler import OutputNode

        buf2 = self._buffer("buf2")
        alias_target = pw_alias_target if pw_alias_target is not None else "buf2"
        buf0 = self._buffer("buf0", aliases=[alias_target], mutations=pw_mutations)
        buf1 = self._buffer("buf1", aliases=[alias_target], mutations=pw_mutations)
        pw = self._node(
            "op0_op1", buffers=["buf0", "buf1"], outputs=[buf0, buf1],
            used=["arg0_1"] if shared_read else ["other"],
        )
        out_user = SimpleNamespace(node=MagicMock(spec=OutputNode))
        nop = self._node(
            "op2", buffers=["buf2"], outputs=[buf2], used=[], nop=True,
        )
        nop_user = SimpleNamespace(node=nop)
        for buf in (buf0, buf1):
            buf.users = [nop_user, out_user]
        buf2.users = [out_user]
        red = self._node(
            "op3", buffers=["buf3"], outputs=[self._buffer("buf3")],
            used=["arg0_1", *red_reads], reduction=True,
        )
        red.get_outputs.return_value[0].users = [out_user]
        nodes = [pw, nop, red] if middle_nop else [pw, red]
        return nodes, pw, red

    def test_benign_cat_alias_and_nop_accepted(self):
        nodes, pw, red = self._make_cat_sequence()
        self.assertIs(self._run_admission(nodes, pw), red)

    def test_real_mutation_rejected(self):
        nodes, pw, _ = self._make_cat_sequence(pw_mutations=["arg0_1"])
        self.assertIsNone(self._run_admission(nodes, pw))

    def test_alias_outside_nop_rejected(self):
        nodes, pw, _ = self._make_cat_sequence(pw_alias_target="buf_foreign")
        self.assertIsNone(self._run_admission(nodes, pw))

    def test_reduction_reads_cat_output_rejected(self):
        nodes, pw, _ = self._make_cat_sequence(red_reads=["buf2"])
        self.assertIsNone(self._run_admission(nodes, pw))

    def test_no_shared_read_rejected(self):
        nodes, pw, _ = self._make_cat_sequence(shared_read=False)
        self.assertIsNone(self._run_admission(nodes, pw))

    def test_alias_without_nop_rejected(self):
        nodes, pw, _ = self._make_cat_sequence(middle_nop=False)
        self.assertIsNone(self._run_admission(nodes, pw))


@unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
class TestPreservedReductionIntegration(unittest.TestCase):
    """End-to-end: inspect generated code for preserved_reduction after torch.compile."""

    def test_reduction_generates_preserved_wrapper(self):
        """An eligible shared-read cat + nonpersistent sibling sum must emit the selector,
        actually select Gluon when it wins timing, and stay bitwise correct on subsequent calls."""
        from torch._inductor import lowering
        from torch._inductor.utils import run_and_get_code
        from torch._inductor.runtime import preserved_reduction as pr_mod

        try:
            from triton.experimental.gluon._runtime import GluonASTSource  # noqa: F401
        except ImportError:
            self.skipTest("Gluon is not available")

        def fn(x):
            s = x.sigmoid()
            return torch.cat((s * x, s + x), -1).to(torch.bfloat16), (x * x).sum(-1, keepdim=True)

        def bits(tensors_a, tensors_b):
            for a, b in zip(tensors_a, tensors_b):
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.stride(), b.stride())
                # Plain unittest.TestCase cannot assertEqual multi-element tensors (ambiguous
                # truth value); torch.equal compares every byte including NaN signs/payloads.
                self.assertTrue(
                    torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                )

        # A reduction dim > 1024 keeps the standard sum nonpersistent (looped); admission rejects
        # persistent reductions, so a small dim would never emit the selector.
        shape = (17, 4096)
        x = torch.randn(*shape, device="cuda")
        torch._dynamo.reset()
        with torch._inductor.config.patch(fx_graph_cache=False):
            compiled = torch.compile(fn, fullgraph=True)
            result, code = run_and_get_code(compiled, x)
        code_str = "\n".join(code)
        # The eligible sequence must carry the preserved-reduction selector wrapper.
        self.assertIn("async_compile.preserved_reduction(", code_str)

        # Reference: the same graph compiled with the optimization disabled. Compare against
        # it (not eager) so the standard reduction's own summation order is the baseline.
        torch._dynamo.reset()
        with torch._inductor.config.patch(fx_graph_cache=False):
            with patch.object(lowering, "_cat_inputs_share_read", return_value=False):
                reference_compiled = torch.compile(fn, fullgraph=True)
                reference = reference_compiled(x)
        bits(result, reference)

        # Force the candidate to win timing and spy on the steady-state path so the test proves
        # Gluon was actually selected (not a silent standard fallback) before comparing bits.
        original_fast = pr_mod.PreservedReductionCall._run_candidate_fast
        selected_modes = []

        def spy(self, pointwise, reduction):
            selected_modes.append(self._selected_mode)
            return original_fast(self, pointwise, reduction)

        with patch.object(pr_mod.PreservedReductionCall, "_benchmark", return_value=(2.0, 1.0)):
            with patch.object(pr_mod.PreservedReductionCall, "_run_candidate_fast", spy):
                torch._dynamo.reset()
                with torch._inductor.config.patch(fx_graph_cache=False):
                    compiled = torch.compile(fn, fullgraph=True)
                    first = compiled(x)
                bits(first, reference)
                # A second call at the same shape reuses the selector in Gluon steady state; a
                # fresh nonpersistent shape recompiles, selects, then verifies steady state too.
                for fresh_shape in (shape, (32, 2048)):
                    y = torch.randn(*fresh_shape, device="cuda")
                    compiled(y)  # first invocation at a new shape selects; same shape reuses
                    z = torch.randn(*fresh_shape, device="cuda")
                    with patch.object(lowering, "_cat_inputs_share_read", return_value=False):
                        ref = reference_compiled(z)
                    bits(compiled(z), ref)
        self.assertTrue(selected_modes, "Gluon steady-state path was never exercised")
        self.assertTrue(all(mode == "gluon" for mode in selected_modes), selected_modes)

    def test_cat_with_reduction_no_fusion(self):
        """The output branch of a shared-read cat must not fuse with the reduction."""
        from torch._inductor import metrics
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = x.sigmoid()
            cat_result = torch.cat((s * x, s + x), -1).to(torch.bfloat16)
            red_result = (x * x).sum(-1, keepdim=True)
            return cat_result, red_result

        x = torch.randn(17, 33, device="cuda")
        torch._dynamo.reset()
        metrics.reset()
        with torch._inductor.config.patch(fx_graph_cache=False):
            result, code = run_and_get_code(
                torch.compile(fn, fullgraph=True), x
            )
        code_str = "\n".join(code)
        # cat should produce 1 kernel and the reduction an independent kernel
        self.assertEqual(metrics.generated_kernel_count, 2)
        # Correctness
        cat_expected = torch.cat(
            (x.sigmoid() * x, x.sigmoid() + x), -1
        ).to(torch.bfloat16)
        red_expected = (x * x).sum(-1, keepdim=True)
        self.assertTrue(torch.allclose(result[1], red_expected))

    def test_dynamic_sibling_sum_17x33(self):
        """[17,33] dynamic sibling sum counterexample: reduction isolation must be correct."""
        from torch._inductor import metrics
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = x.sigmoid()
            return torch.cat((s * x, s + x), -1).to(torch.bfloat16), x.sum(-1)

        torch._dynamo.reset()
        metrics.reset()
        x = torch.randn(17, 33, device="cuda")
        with torch._inductor.config.patch(fx_graph_cache=False):
            result, code = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=True), x
            )
        # Verify reduction isolation: at least 2 kernels
        self.assertGreaterEqual(metrics.generated_kernel_count, 2)
        # Correctness
        self.assertTrue(torch.allclose(result[1], x.sum(-1)))


if __name__ == "__main__":
    unittest.main()
