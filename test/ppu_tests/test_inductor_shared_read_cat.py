# Owner(s): ["module: inductor"]
from unittest.mock import patch

import torch
import torch._inductor.config as config
from torch._inductor import lowering, metrics
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_gpu_and_triton


aten = torch.ops.aten
DEVICE = GPU_TYPE


def T(*size, dtype=torch.float32, device=DEVICE, grad=False):
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)


class TestCase(InductorTestCase):
    device = DEVICE


class SharedReadCatTests(TestCase):
    def compile_cat(self, fn, args, *, force_pointwise=False, dynamic=False):
        torch._dynamo.reset()
        metrics.reset()
        with config.patch(force_pointwise_cat=force_pointwise, fx_graph_cache=False):
            result, code = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=dynamic), *args
            )
        return result, "\n".join(code), metrics.generated_kernel_count

    @requires_gpu_and_triton
    def test_shared_reads(self):
        for shape, dim in [
            ((257,), 0),
            ((17, 33), 0),
            ((17, 33), -1),
            ((3, 1, 65), -1),
        ]:
            with self.subTest(shape=shape, dim=dim):

                def fn(x, y):
                    s = x.sigmoid()
                    return torch.cat((s * y, s + x), dim)

                args = (T(*shape), T(*shape))
                result, code, kernels = self.compile_cat(fn, args)
                self.assertEqual(result, fn(*args))
                self.assertEqual(kernels, 1)
                self.assertEqual(code.count("tl.load("), 2)
                self.assertEqual(code.count("tl.store("), 2)
                self.assertEqual(code.count("tl.sigmoid("), 1)
                reference, old_code, old_kernels = self.compile_cat(
                    fn, args, force_pointwise=True
                )
                self.assertEqual(result, reference, atol=0, rtol=0)
                self.assertEqual(old_kernels, 1)
                self.assertEqual(old_code.count("tl.store("), 1)

    @requires_gpu_and_triton
    def test_fallbacks(self):
        x, y, b = T(17, 33), T(17, 33), T(33)

        def multi_user(x):
            a = x.sin()
            return torch.cat((a, x.cos()), -1), a

        cases = [
            (
                "different_inputs",
                lambda x, y: torch.cat((x.sin(), y.cos()), -1),
                (x, y),
                False,
            ),
            (
                "different_offsets",
                lambda x: torch.cat((x[:, :16].sin(), x[:, 16:32].cos()), -1),
                (x,),
                False,
            ),
            (
                "different_shapes",
                lambda x: torch.cat((x[:, :16].sin(), x[:, 16:].cos()), -1),
                (x,),
                False,
            ),
            (
                "different_indices",
                lambda x: torch.cat((x.sin(), x.t().cos()), -1),
                (T(17, 17),),
                False,
            ),
            (
                "downstream",
                lambda x: torch.cat((x.sin(), x.cos()), -1).cos(),
                (x,),
                False,
            ),
            ("multiple_users", multi_user, (x,), False),
            (
                "broadcast_only",
                lambda x, y, b: torch.cat((x + b, y * b), -1),
                (x, y, b),
                False,
            ),
            (
                "reduction",
                lambda x: torch.cat((x.sum(-1).sin(), x.sum(-1).cos()), -1),
                (x,),
                False,
            ),
            (
                "realized_input",
                lambda x: torch.cat((x, x.cos()), -1),
                (x,),
                False,
            ),
            (
                "integer_conversion",
                lambda x: torch.cat((x.sin(), x.cos()), -1).to(torch.int32),
                (x,),
                False,
            ),
            (
                "conversion_downstream",
                lambda x: torch.cat((x.sin(), x.cos()), -1)
                .to(torch.float16)
                .cos(),
                (x,),
                False,
            ),
            (
                "cat_multiple_users",
                lambda x: (
                    (z := torch.cat((x.sin(), x.cos()), -1)),
                    z.to(torch.bfloat16),
                ),
                (x,),
                True,
            ),
        ]
        original = lowering._cat_inputs_share_read
        for name, fn, args, dynamic in cases:
            with self.subTest(name=name):
                decisions = []

                def record(inputs):
                    decision = original(inputs)
                    decisions.append(decision)
                    return decision

                with patch.object(lowering, "_cat_inputs_share_read", record):
                    result, _, _ = self.compile_cat(fn, args, dynamic=dynamic)
                self.assertFalse(any(decisions), decisions)
                self.assertEqual(result, fn(*args))

    def assert_same_bits(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(torch.isnan(actual), torch.isnan(expected))
        # Compare every byte including NaN signs and payloads; masking NaNs out would
        # let NaN-bit changes pass unnoticed.
        self.assertEqual(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )

    @requires_gpu_and_triton
    def test_dynamic_terminal_cast(self):
        for cast in (False, True):
            with self.subTest(cast=cast):
                torch._dynamo.reset()
                decisions = []
                graphs = []
                original = lowering._cat_inputs_share_read

                def fn(x):
                    s = x.sigmoid()
                    result = torch.cat((s * x, s + x), -1)
                    return result.to(torch.bfloat16) if cast else result

                def record(inputs):
                    decision = original(inputs)
                    decisions.append(decision)
                    return decision

                def backend(gm, inputs):
                    with patch.object(lowering, "_cat_inputs_share_read", record):
                        from torch._inductor.graph import GraphLowering

                        with patch.object(GraphLowering, "save_output_code", graphs.append):
                            return torch._inductor.compile(gm, inputs)

                with config.patch(fx_graph_cache=False):
                    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
                    for shape in ((17, 33), (19, 65), (31, 33)):
                        x = T(*shape)
                        actual = compiled(x)
                        self.assertEqual(actual, fn(x))
                self.assertTrue(decisions)
                self.assertTrue(all(decisions), decisions)
                self.assertTrue(graphs)
                for code in graphs:
                    self.assertEqual(code.count("tl.load("), 1)
                    self.assertEqual(code.count("tl.store("), 2)

    @requires_gpu_and_triton
    def test_terminal_cast_numerics(self):
        def fn(x, target):
            return torch.cat((x * 1.125 + 0.25, x * 0.875 - 0.25), -1).to(target)

        for source in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for target in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                with self.subTest(source=source, target=target):
                    x = T(17, 33, dtype=source)
                    x.flatten()[:8] = torch.tensor(
                        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), -80, 80, 1e-7],
                        device=DEVICE,
                        dtype=source,
                    )
                    actual, code, kernels = self.compile_cat(fn, (x, target))
                    with patch.object(lowering, "_cat_inputs_share_read", return_value=False):
                        expected, _, _ = self.compile_cat(fn, (x, target))
                    self.assert_same_bits(actual, expected)
                    self.assertEqual(kernels, 1)
                    self.assertEqual(code.count("tl.store("), 2)

        for source in (torch.float16, torch.bfloat16):
            x = torch.arange(65536, device=DEVICE, dtype=torch.int32).to(torch.int16)
            x = x.view(source).reshape(64, 1024)
            actual, _, _ = self.compile_cat(fn, (x, torch.bfloat16))
            with patch.object(lowering, "_cat_inputs_share_read", return_value=False):
                expected, _, _ = self.compile_cat(fn, (x, torch.bfloat16))
            self.assert_same_bits(actual, expected)

    @requires_gpu_and_triton
    def test_type_promotion_terminal_cast(self):
        def fn(x):
            # Existing rounding on the FP16 branch must precede the FP32 type promotion of cat.
            return torch.cat((x * 1.125 + 0.25, x.float() * 0.875 - 0.25), -1).to(torch.bfloat16)

        x = T(17, 33, dtype=torch.float16)
        actual, code, kernels = self.compile_cat(fn, (x,))
        with patch.object(lowering, "_cat_inputs_share_read", return_value=False):
            expected, _, _ = self.compile_cat(fn, (x,))
        self.assert_same_bits(actual, expected)
        self.assertEqual(kernels, 1)
        self.assertEqual(code.count("tl.store("), 2)

    @requires_gpu_and_triton
    def test_terminal_cast_skips_heuristic_realization(self):
        def fn(g, y):
            a, b = y.chunk(2, -1)
            s = a.sigmoid()
            return torch.cat((g * s * (1 + a * (1 - s)) * b, g * s * a), -1).to(
                torch.bfloat16
            )

        args = (T(17, 33), T(17, 66))
        with config.patch(realize_opcount_threshold=15):
            actual, code, kernels = self.compile_cat(fn, args)
            with patch.object(lowering, "_cat_inputs_share_read", return_value=False):
                expected, _, _ = self.compile_cat(fn, args)
        self.assert_same_bits(actual, expected)
        self.assertEqual(kernels, 1)
        self.assertEqual(code.count("tl.store("), 2)

    @requires_gpu_and_triton
    def test_terminal_cast_independent_reduction(self):
        def fn(x):
            s = x.sigmoid()
            return torch.cat((s * x, s + x), -1).to(torch.bfloat16), (x * x).sum(-1, keepdim=True)

        for dynamic in (False, True):
            for shape in ((17, 33), (1024, 512), (17, 4096)):
                with self.subTest(dynamic=dynamic, shape=shape):
                    x = T(*shape)
                    actual, code, kernels = self.compile_cat(fn, (x,), dynamic=dynamic)
                    with patch.object(lowering, "_cat_inputs_share_read", return_value=False):
                        expected, _, _ = self.compile_cat(fn, (x,), dynamic=dynamic)
                    for a, b in zip(actual, expected):
                        self.assert_same_bits(a, b)
                    # The reduction stays independent: the shared-read cat branch is never fused
                    # with the sibling reduction, so exactly two standard kernels are emitted.
                    self.assertEqual(kernels, 2)
                    self.assertEqual(code.count("tl.sum("), 1)
                    self.assertEqual(code.count("tl.store("), 3)

    @requires_gpu_and_triton
    def test_swiglu_numerics(self):
        def backward(g, y):
            a, b = y.chunk(2, -1)
            return torch.cat(
                (
                    g * a.sigmoid() * (1 + a * (1 - a.sigmoid())) * b,
                    g * torch.nn.functional.silu(a),
                ),
                -1,
            )

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = T(17, 1, 66, dtype=dtype)
                g = T(17, 1, 33, dtype=dtype)
                x.flatten()[:8] = torch.tensor(
                    [0.0, -0.0, float("inf"), -float("inf"), float("nan"), -80, 80, 1e-7],
                    device=DEVICE,
                    dtype=dtype,
                )
                result, code, kernels = self.compile_cat(backward, (g, x))
                reference, _, _ = self.compile_cat(
                    backward, (g, x), force_pointwise=True
                )
                self.assertEqual(kernels, 1)
                self.assertEqual(code.count("tl.load("), 3)
                self.assertEqual(code.count("tl.store("), 2)
                self.assertEqual(torch.isnan(result), torch.isnan(reference))
                bits = torch.int32 if dtype == torch.float32 else torch.int16
                self.assertEqual(result.view(bits), reference.view(bits))

        # Exhaust all BF16 encodings, including subnormals, infinities, NaNs and signed zeros.
        a = torch.arange(65536, device=DEVICE, dtype=torch.int32).to(torch.int16)
        a = a.view(torch.bfloat16).reshape(64, 1, 1024)
        b = T(64, 1, 1024, dtype=torch.bfloat16)
        g = T(64, 1, 1024, dtype=torch.bfloat16)
        args = (g, torch.cat((a, b), -1))
        result, code, kernels = self.compile_cat(backward, args)
        reference, _, _ = self.compile_cat(backward, args, force_pointwise=True)
        self.assertEqual(kernels, 1)
        self.assertEqual(code.count("tl.store("), 2)
        self.assertEqual(torch.isnan(result), torch.isnan(reference))
        self.assertEqual(result.view(torch.int16), reference.view(torch.int16))

    @requires_gpu_and_triton
    def test_autograd(self):
        def fn(x):
            return torch.cat((x.sin(), x.cos()), -1)

        x = T(17, 33, grad=True)
        actual = torch.compile(fn, fullgraph=True)(x)
        expected = fn(x)
        grad = torch.randn_like(expected)
        self.assertEqual(actual, expected)
        self.assertEqual(
            torch.autograd.grad(actual, x, grad),
            torch.autograd.grad(expected, x, grad),
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU_AND_TRITON:
        run_tests(needs="filelock")
