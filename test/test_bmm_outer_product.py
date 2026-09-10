# Owner(s): ["module: nn"]

import unittest

import torch
from torch._native.ops.bmm_outer_product.triton_impl import _is_outer_product
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import run_tests, skipIfXpu, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


@unittest.skipIf(not HAS_GPU, "requires GPU")
class TestBmmOuterProduct(TestCase):
    def _check_bmm(self, a, b, **kwargs):
        self.assertEqual(torch.bmm(a, b), a @ b, **kwargs)

    @skipIfXpu(msg="https://github.com/pytorch/pytorch/issues/180318")
    def test_shapes(self):
        shapes = [
            (4, 8, 16),
            (32, 8, 256),
            (16, 128, 512),
            (1, 64, 128),
            (8, 1, 1),
            (64, 256, 512),
            (256, 8, 2048),
        ]
        for B, M, N in shapes:
            with self.subTest(B=B, M=M, N=N):
                a = torch.randn(B, M, 1, device=GPU_TYPE)
                b = torch.randn(B, 1, N, device=GPU_TYPE)
                self._check_bmm(a, b)

    def test_basic_dtypes(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                a = torch.randn(4, 8, 1, device=GPU_TYPE, dtype=dtype)
                b = torch.randn(4, 1, 16, device=GPU_TYPE, dtype=dtype)
                self.assertEqual(torch.bmm(a, b), a @ b)

    def test_permuted_inputs(self):
        B, M, N = 4, 8, 16
        cases = [
            (
                torch.randn(M, B, 1, device=GPU_TYPE).permute(1, 0, 2),
                torch.randn(B, 1, N, device=GPU_TYPE),
            ),
            (
                torch.randn(B, M, 1, device=GPU_TYPE),
                torch.randn(N, B, 1, device=GPU_TYPE).permute(1, 2, 0),
            ),
            (
                torch.randn(M, B, 1, device=GPU_TYPE).permute(1, 0, 2),
                torch.randn(N, B, 1, device=GPU_TYPE).permute(1, 2, 0),
            ),
        ]
        for a, b in cases:
            self.assertEqual(torch.bmm(a, b), a @ b)

    def test_fallback_non_outer_product(self):
        a = torch.randn(4, 8, 16, device=GPU_TYPE)
        b = torch.randn(4, 16, 32, device=GPU_TYPE)
        self.assertEqual(torch.bmm(a, b), a @ b, atol=1e-5, rtol=1.3e-6)

    def test_batch_one(self):
        a = torch.randn(1, 64, 1, device=GPU_TYPE)
        b = torch.randn(1, 1, 128, device=GPU_TYPE)
        self.assertEqual(torch.bmm(a, b), a @ b)

    @largeTensorTest("6GB")
    def test_batch_offset_past_int32_max(self):
        # The Triton kernel computed its element offsets in int32. With
        # (batch, M, N) = (512, 8209, 512) the last batch matrix starts at
        # 511 * 8209 * 512 = 2_147_737_088 > INT32_MAX, so those batches were
        # stored gigabytes before the output buffer: a CUDA fault, or, when
        # that address was live memory, uninitialised rows read back for the
        # overflowing batches. Either way the comparison below cannot pass on
        # the old kernel. batch = 8208 rows was the last size that worked.
        # Strided views exercise the same arithmetic on the input loads.
        batch, m, n = 512, 8209, 512
        self.assertGreater((batch - 1) * m * n, torch.iinfo(torch.int32).max)
        for strided in (False, True):
            with self.subTest(strided=strided):
                a = torch.randn(batch, m, 2, device=GPU_TYPE, dtype=torch.bfloat16)
                b = torch.randn(batch, 2, n, device=GPU_TYPE, dtype=torch.bfloat16)
                if strided:
                    a, b = a[:, :, :1], b[:, :1, :]
                else:
                    a, b = a[:, :, :1].contiguous(), b[:, :1, :].contiguous()
                out = torch.bmm(a, b)
                # The broadcast product is the reference: `a @ b` would
                # dispatch to the kernel under test, and a full reference
                # would be another 4 GiB.
                for i in (0, batch - 2, batch - 1):
                    self.assertEqual(out[i], a[i] * b[i])
                del out

    @largeTensorTest("6GB")
    def test_row_offset_past_int32_max(self):
        # Inside one batch matrix the store offset is rm * stride_om, so with
        # (M - 1) * N > INT32_MAX (M, N themselves int32) the last row tiles
        # wrapped as well: (65536 + 4096, 32768) puts the last 4096 rows past
        # the limit in a 4.25 GiB bf16 output. (M > INT32_MAX on its own is
        # not reachable: Triton then specialises M as int64 and every index
        # derived from it is already 64-bit.)
        m, n = 2**16 + 4096, 2**15
        self.assertGreater((m - 1) * n, torch.iinfo(torch.int32).max)
        a = torch.randn(1, m, 1, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(1, 1, n, device=GPU_TYPE, dtype=torch.bfloat16)
        out = torch.bmm(a, b)
        for rows in (
            slice(0, 1024),
            slice(2**16 - 1024, 2**16 + 1024),
            slice(m - 1024, m),
        ):
            self.assertEqual(out[0, rows], a[0, rows] * b[0])

    def test_m_one_n_one(self):
        a = torch.randn(8, 1, 1, device=GPU_TYPE)
        b = torch.randn(8, 1, 1, device=GPU_TYPE)
        self.assertEqual(torch.bmm(a, b), a @ b)

    def test_gradient_flow(self):
        a = torch.randn(4, 8, 1, device=GPU_TYPE, requires_grad=True)
        b = torch.randn(4, 1, 16, device=GPU_TYPE, requires_grad=True)
        result = torch.bmm(a, b)
        result.sum().backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertEqual(a.grad.shape, a.shape)
        self.assertEqual(b.grad.shape, b.shape)

    def test_cpu_outer_product_fallback(self):
        a = torch.randn(4, 8, 1)
        b = torch.randn(4, 1, 16)
        self.assertTrue(a.device.type == "cpu")
        self.assertEqual(torch.bmm(a, b), a @ b)

    def test_mixed_device_outer_product_fallback(self):
        a = torch.randn(4, 8, 1)
        b = torch.randn(4, 1, 16, device=GPU_TYPE)
        with self.assertRaises(RuntimeError):
            torch.bmm(a, b)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_non_current_device_outer_product(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("requires at least 2 visible CUDA devices")

        old_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            a = torch.randn(4, 8, 1, device="cuda:1")
            b = torch.randn(4, 1, 16, device="cuda:1")

            out = torch.bmm(a, b)

            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(out.device, torch.device("cuda:1"))
            self.assertEqual(out, a * b)

            mismatched_a = torch.randn(4, 8, 1, device="cuda:0")
            with self.assertRaisesRegex(RuntimeError, "same device|different"):
                torch.bmm(mismatched_a, b)
        finally:
            torch.cuda.set_device(old_device)


class TestOuterProductDetection(TestCase):
    def test_is_outer_product(self):
        self.assertTrue(_is_outer_product(torch.empty(4, 8, 1), torch.empty(4, 1, 16)))
        self.assertTrue(_is_outer_product(torch.empty(4, 8, 1), torch.empty(4, 1, 1)))
        self.assertFalse(
            _is_outer_product(torch.empty(4, 8, 16), torch.empty(4, 16, 32))
        )
        self.assertFalse(_is_outer_product(torch.empty(8, 1), torch.empty(1, 16)))
        self.assertFalse(_is_outer_product(torch.empty(4, 8, 1), torch.empty(4, 2, 16)))
        self.assertFalse(_is_outer_product(torch.empty(4, 8, 3), torch.empty(4, 1, 16)))
        self.assertFalse(
            _is_outer_product(
                torch.empty(4, 8, 1, dtype=torch.complex64),
                torch.empty(4, 1, 16, dtype=torch.complex64),
            )
        )


if __name__ == "__main__":
    run_tests()
