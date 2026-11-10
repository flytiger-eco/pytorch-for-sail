# Owner(s): ["module: scatter & gather ops"]

import operator
import random
from functools import reduce

import torch

from torch.testing import make_tensor
from torch.testing._internal.common_utils import \
    (parametrize, run_tests, TestCase, DeterministicGuard, TEST_WITH_ROCM, serialTest)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, onlyCPU, onlyCUDA, dtypes, dtypesIfCUDA,
     largeTensorTest, toleranceOverride, tol,)
from torch.testing._internal.common_dtype import \
    (get_all_dtypes,)

from torch.testing._internal.common_cuda import CDNA3OrLater

# Protects against includes accidentally setting the default dtype
if torch.get_default_dtype() is not torch.float32:
    raise AssertionError("default dtype should be float32")


# Note: test_scatter_gather_ops.py
# This test file tests scatter and gather operations,
#   like torch.scatter and torch.gather.

class TestScatterGather(TestCase):
    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("shape", [(), (7,), (7, 6), (3, 7, 5), (2, 3, 7, 4),
                           (2, 1, 3, 7, 3), (1,) * 24])
    @parametrize("index_layout", ["broadcast", "dense", "strided", "permuted"])
    @parametrize("partial", [False, True])
    @parametrize("offsets", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_scatter_add_pairs_nd(self, device, dtype, shape, index_layout, partial, offsets):
        rank = len(shape)
        dims = (0, rank // 2, rank - 1) if rank > 5 else range(max(1, rank))
        for d in dims:
            output_shape, source_shape = list(shape), list(shape)
            if rank:
                output_shape[d], source_shape[d] = 7, 11
                if partial:
                    source_shape = [s + 2 if i != d and s > 1 else s
                                    for i, s in enumerate(source_shape)]
            index_shape = list(source_shape)
            if partial and rank:
                index_shape = [9 if i == d else max(1, output_shape[i] - 1)
                               for i in range(rank)]
            source_numel = reduce(operator.mul, source_shape, 1)
            output_numel = reduce(operator.mul, output_shape, 1)
            index_numel = reduce(operator.mul, index_shape, 1)
            source_cpu = ((torch.arange(source_numel).reshape(source_shape) % 7 - 3).float() * 0.125).to(dtype)
            initial = torch.full(output_shape, 0.125, dtype=dtype)
            if rank and index_layout == "broadcast":
                index_view = [1] * rank
                index_view[d] = index_shape[d]
                ids = (torch.arange(index_shape[d]) * 3) % 7
                index_cpu = ids.reshape(index_view).expand(index_shape)
            else:
                index_cpu = ((torch.arange(index_numel) * 3) % 7).reshape(index_shape)
            expected = initial.float().scatter_add(d, index_cpu, source_cpu.float()).to(dtype)
            for dim in (d, d - max(1, rank)):
                output_offset, source_offset = offsets
                source_storage = torch.full((source_numel + 4,), 17.0, device=device, dtype=dtype)
                source = source_storage[source_offset:source_offset + source_numel].reshape(source_shape)
                source.copy_(source_cpu)
                preserved_source = source_storage.clone()
                output_storage = torch.full((output_numel + 4,), 9.0, device=device, dtype=dtype)
                output = output_storage[output_offset:output_offset + output_numel].reshape(output_shape)
                output.copy_(initial)
                if rank and index_layout == "broadcast":
                    index = ids.to(device).reshape(index_view).expand(index_shape)
                else:
                    index = index_cpu.to(device)
                    if rank and index_layout == "strided":
                        index = index.repeat_interleave(2, dim=-1)[..., ::2]
                    elif rank > 1 and index_layout == "permuted":
                        index = index.transpose(0, -1).contiguous().transpose(0, -1)
                with DeterministicGuard(False):
                    output.scatter_add_(dim, index, source)
                    functional = torch.scatter_add(initial.to(device), dim, index, source)
                    out = torch.empty_like(output)
                    torch.scatter_add(initial.to(device), dim, index, source, out=out)
                bits = torch.int32 if dtype == torch.float32 else torch.int16
                for actual in (output, functional, out):
                    self.assertEqual(actual.cpu().view(bits), expected.view(bits))
                self.assertEqual(source_storage, preserved_source)
                self.assertEqual(output_storage[:output_offset], torch.full_like(output_storage[:output_offset], 9.0))
                tail = output_storage[output_offset + output_numel:]
                self.assertEqual(tail, torch.full_like(tail, 9.0))

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("shape, dim", [((7,), 0), ((3, 7, 5), 1), ((2, 3, 7, 4), 2),
                                ((3, 7, 5), -1), ((1,) * 12 + (7,) + (1,) * 11, 12)])
    @parametrize("deterministic", [False, True])
    def test_gather_pair_backward_nd(self, device, dtype, shape, dim, deterministic):
        d = dim % len(shape)
        grad_shape, index_view = list(shape), [1] * len(shape)
        grad_shape[d], index_view[d] = 11, 11
        ids = (torch.arange(11) * 3) % shape[d]
        index_cpu = ids.reshape(index_view).expand(grad_shape)
        numel = reduce(operator.mul, grad_shape, 1)
        grad_cpu = ((torch.arange(numel).reshape(grad_shape) % 7 - 3).float() * 0.125).to(dtype)
        expected = torch.zeros(shape).scatter_add(dim, index_cpu, grad_cpu.float()).to(dtype)
        leaf = torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)
        index = ids.to(device).reshape(index_view).expand(grad_shape)
        with DeterministicGuard(deterministic):
            leaf.gather(dim, index).backward(grad_cpu.to(device))
        self.assertEqual(leaf.grad.cpu().view(torch.int16), expected.view(torch.int16))

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_layout", ["broadcast", "dense", "strided", "partial"])
    @parametrize("deterministic", [False, True])
    @parametrize("offset", [0, 1])
    def test_scatter_add_pair_nd_dispatch(self, device, dtype, index_layout, deterministic, offset):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        shape = (3, 7, 5)
        source_shape = (5, 11, 7) if index_layout == "partial" else (3, 11, 5)
        index_shape = (2, 9, 4) if index_layout == "partial" else source_shape
        source = torch.full(source_shape, 0.125, device=device, dtype=dtype)
        storage = torch.zeros(3 * 7 * 5 + 2, device=device, dtype=dtype)
        output = storage[offset:offset + 3 * 7 * 5].reshape(shape)
        if index_layout == "broadcast":
            index = (torch.arange(11, device=device) % 7).reshape(1, 11, 1).expand(index_shape)
        else:
            numel = reduce(operator.mul, index_shape, 1)
            index = (torch.arange(numel, device=device) % 7).reshape(index_shape)
            if index_layout == "strided":
                index = index.repeat_interleave(2, dim=-1)[..., ::2]
        with DeterministicGuard(deterministic), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            output.scatter_add_(-2, index, source)
        expected = torch.zeros(shape).scatter_add(1, index.cpu(), source.float().cpu()).to(dtype)
        self.assertEqual(output.cpu().view(torch.int16), expected.view(torch.int16))
        kernels = [event.name for event in prof.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        hit = any("ppu_index_scatter_opt::pair_atomic_add_" in name for name in kernels)
        self.assertFalse(hit)
        if not deterministic:
            self.assertTrue(any("scatter_gather" in name for name in kernels))

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @serialTest()
    @largeTensorTest("5GB")
    def test_scatter_add_pairs_nd_64bit_addressing(self, device, dtype):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        size = (1 << 31) + 5
        output = torch.zeros((1, size, 1), device=device, dtype=dtype)
        source = torch.full((2, 7, 1), 0.125, device=device, dtype=dtype)
        index = torch.tensor([size - 1, size - 2, 0, 1, 2, 3, 4], device=device).reshape(1, 7, 1)
        with DeterministicGuard(False), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            output.scatter_add_(-2, index, source)
        self.assertEqual(output.count_nonzero().item(), 7)
        self.assertEqual(output[0, index.reshape(-1), 0], torch.full((7,), 0.125, device=device, dtype=dtype))
        kernels = [event.name for event in prof.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        self.assertFalse(any("ppu_index_scatter_opt::pair_atomic_add_" in name for name in kernels))
        self.assertTrue(any("scatter_gather" in name for name in kernels))

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("shape, dim", [((11, 8), 0), ((11, 3, 8), 0), ((3, 11, 8), 1), ((11, 7), 0)])
    @parametrize("index_layout", ["broadcast", "dense_same", "dense_mixed", "strided", "partial"])
    @parametrize("offset", [0, 1])
    @parametrize("deterministic", [False, True])
    def test_scatter_add_pair_fastpath_dispatch(self, device, dtype, shape, dim, index_layout, offset, deterministic):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        output_shape = list(shape)
        output_shape[dim] = 7
        source = torch.full(shape, 0.125, device=device, dtype=dtype)
        storage = torch.full((source.numel() + 4,), 9.0, device=device, dtype=dtype)
        source = storage[offset:offset + source.numel()].reshape(shape).copy_(source)
        output_storage = torch.full((reduce(operator.mul, output_shape, 1) + 4,), 9.0, device=device, dtype=dtype)
        output = output_storage[offset:offset + reduce(operator.mul, output_shape, 1)].reshape(output_shape).zero_()
        ids_shape = [1] * len(shape)
        ids_shape[dim] = shape[dim]
        index = (torch.arange(shape[dim], device=device) * 3 % 7).reshape(ids_shape).expand(shape)
        if index_layout != "broadcast":
            index = index.contiguous()
        if index_layout == "dense_mixed":
            index = (index + torch.arange(index.numel(), device=device).reshape(shape)) % 7
        elif index_layout == "strided":
            index = index.repeat_interleave(2, dim=-1)[..., ::2]
        elif index_layout == "partial":
            index = index.narrow(dim, 0, 9)
        expected = torch.zeros(output_shape).scatter_add(dim, index.cpu(), source.float().cpu()).to(dtype)
        source_before = storage.clone()
        with DeterministicGuard(deterministic), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            output.scatter_add_(dim, index, source)
            torch.cuda.synchronize()
        self.assertEqual(output.cpu().view(torch.int32 if dtype == torch.float32 else torch.int16),
                         expected.view(torch.int32 if dtype == torch.float32 else torch.int16))
        self.assertEqual(storage, source_before)
        self.assertEqual(output_storage[:offset], torch.full_like(output_storage[:offset], 9.0))
        tail = output_storage[offset + output.numel():]
        self.assertEqual(tail, torch.full_like(tail, 9.0))
        kernels = [event.name for event in prof.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        hit = any("ppu_index_scatter_opt::pair_atomic_add_kernel" in name for name in kernels)
        inner = reduce(operator.mul, shape[dim + 1:], 1)
        expected_hit = (dtype != torch.float32 and not deterministic and dim == 0 and inner % 2 == 0 and
                        offset == 0 and index_layout in ("broadcast", "dense_same", "dense_mixed"))
        self.assertEqual(hit, expected_hit)
        if not expected_hit and not deterministic:
            self.assertTrue(any("scatter_gather" in name for name in kernels))

    # Fills an index tensor with valid indices
    def _fill_indices(self, idx, dim, dim_size, elems_per_row, m, n, o, unique_indices=True):
        for i in range(1 if dim == 0 else m):
            for j in range(1 if dim == 1 else n):
                for k in range(1 if dim == 2 else o):
                    ii = [i, j, k]
                    ii[dim] = slice(0, idx.size(dim) + 1)
                    if unique_indices:
                        idx[tuple(ii)] = torch.randperm(dim_size)[0:elems_per_row]
                    else:
                        idx[tuple(ii)] = torch.randint(dim_size, (elems_per_row,))

    @dtypes(torch.float32, torch.complex64)
    def test_gather(self, device, dtype):
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        src = make_tensor((m, n, o), device=device, dtype=dtype)
        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = make_tensor(idx_size, device=device, dtype=torch.long)
        self._fill_indices(idx, dim, src.size(dim), elems_per_row, m, n, o)

        actual = torch.gather(src, dim, idx)
        expected = torch.zeros(idx_size, device=device, dtype=dtype)
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    expected[i, j, k] = src[tuple(ii)]
        self.assertEqual(actual, expected, atol=0, rtol=0)

        # Guarded because torch.max isn't defined for complex types
        if not dtype.is_complex:
            src = make_tensor((3, 4, 5), device=device, dtype=dtype)
            expected, idx = src.max(2, True)
            actual = torch.gather(src, 2, idx)
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @serialTest()
    @dtypes(torch.int8, torch.bfloat16)
    def test_gather_large(self, device, dtype):
        # test larger shapes to check vectorized implementation
        for (m, n, k) in ((4096, 3072, 4096), (4096, 3072, 4100), (4, 4, 16384 * 8192)):
            torch.cuda.empty_cache()
            src = make_tensor((m, k), device=device, dtype=dtype)
            alloc0 = torch.empty(src.nelement() * 2, device=device, dtype=dtype)
            discontig = alloc0.view(m, 2 * k)[:, ::2].copy_(src)
            alloc1 = torch.empty(src.nelement() + 1, device=device, dtype=dtype)
            misaligned = alloc1[1:].view(m, k).copy_(src)
            alloc2 = torch.empty(m, k + 4, device=device, dtype=dtype)
            misaligned1 = alloc2[:, :-4].copy_(src)
            num_ind = n
            for dim in (0, 1):
                max_ind = src.shape[dim]
                ind0 = torch.randint(max_ind, (num_ind,), device=device)
                ind_discontig0 = torch.empty(num_ind * 2, device=device, dtype=torch.int64)[::2].copy_(ind0)
                shape_ind = [1] * src.ndim
                shape_ind[dim] = ind0.shape[0]
                shape_out = list(src.shape)
                shape_out[dim] = ind0.shape[0]
                ind = ind0.view(shape_ind).expand(shape_out)
                ind_discontig = ind_discontig0.view(shape_ind).expand(shape_out)
                res = torch.gather(src, dim=dim, index=ind)
                ref = src[ind0] if dim == 0 else src[:, ind0]
                self.assertEqual(res, ref, atol=0, rtol=0)
                if res.device.type == "cuda":
                    ref_cpu = src.cpu()[ind0.cpu()] if dim == 0 else src.cpu()[:, ind0.cpu()]
                    self.assertEqual(res.cpu(), ref_cpu, atol=0, rtol=0)
                res = torch.gather(src, dim=dim, index=ind_discontig)
                self.assertEqual(res, ref, atol=0, rtol=0)
                res_ind = src[ind_discontig0] if dim == 0 else src[:, ind_discontig0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res_ind_neg = src[ind0 - src.shape[dim]] if dim == 0 else src[:, ind0 - src.shape[1]]
                self.assertEqual(res_ind_neg, ref, atol=0, rtol=0)
                res = torch.gather(discontig, dim=dim, index=ind)
                self.assertEqual(res, ref, atol=0, rtol=0)
                res_ind = discontig[ind0] if dim == 0 else discontig[:, ind0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res = torch.gather(misaligned, dim=dim, index=ind)
                self.assertEqual(res, ref, atol=0, rtol=0)
                res_ind = misaligned[ind0] if dim == 0 else misaligned[:, ind0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res_ind = misaligned1[ind0] if dim == 0 else misaligned[:, ind0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res_gather = torch.gather(misaligned1, dim=dim, index=ind)
                self.assertEqual(res_gather, ref, atol=0, rtol=0)
            del src, alloc0, alloc1, alloc2
            del discontig, misaligned, misaligned1
        # test gather along 1st dim that can accidentally trigger fast path
        # because due to index dimension in the gather dim being 1
        # an unexpected squashing in tensorIterator happens
        src = make_tensor((16, 2, 16), device=device, dtype=dtype)
        ind = torch.randint(2, (16, 1), device=device).view(16, 1, 1).expand(16, 1, 16)
        res = torch.gather(src, dim=1, index=ind)
        if res.device.type == "cuda":
            ref_cpu = torch.gather(src.cpu(), dim=1, index=ind.cpu())
            self.assertEqual(res.cpu(), ref_cpu, atol=0, rtol=0)


    @dtypes(torch.bool)
    def test_gather_bool(self, device, dtype):
        src = torch.tensor(((False, True), (True, True)), device=device, dtype=dtype)
        idx = torch.tensor(((0, 0), (1, 0)), device=device, dtype=torch.long)
        actual = torch.gather(src, 1, idx)
        expected = torch.tensor(((False, False), (True, True)), device=device, dtype=dtype)
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @parametrize("sparse_grad", [False, True])
    @dtypes(torch.float32, torch.float64)
    def test_gather_backward_with_empty_index_tensor(self, device, dtype, sparse_grad):
        dim = -1
        input = torch.rand([10, 5], dtype=dtype, device=device, requires_grad=True)
        index = torch.randint(0, 2, [3, 0], dtype=torch.int64, device=device)
        res = torch.gather(input, dim, index, sparse_grad=sparse_grad)
        res.sum().backward()
        grad = input.grad.to_dense() if sparse_grad else input.grad
        expected_grad = torch.zeros_like(input, requires_grad=False)
        self.assertEqual(grad, expected_grad, atol=0, rtol=0)

    def _test_scatter_base(self, fn, *, device, dtype, is_scalar, reduction,
                           unique_indices=True, include_self=True):
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = torch.empty(tuple(idx_size), device=device, dtype=torch.long)
        self._fill_indices(idx, dim, ([m, n, o])[dim], elems_per_row, m, n, o, unique_indices)

        if is_scalar:
            src = random.random()
        else:
            src_size = [random.randint(1, 5) + s for s in idx_size]
            src = make_tensor(tuple(src_size), device=device, dtype=dtype)

        base = make_tensor((m, n, o), device=device, dtype=dtype)
        if reduction is not None:
            if fn is torch.Tensor.scatter_reduce_:
                actual = fn(base.clone(), dim, idx, src, reduce=reduction, include_self=include_self)
            else:
                actual = fn(base.clone(), dim, idx, src, reduce=reduction)
        else:
            actual = fn(base.clone(), dim, idx, src)

        expected = base.clone()
        counts = torch.zeros(base.shape, dtype=torch.long, device=device) + include_self
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    if fn is torch.Tensor.scatter_add_:
                        expected[tuple(ii)] += src[i, j, k]
                    else:
                        # method may be 'scatter_', 'scatter', 'scatter_reduce'
                        # or 'scatter_reduce_', the former two might have a reduction argument
                        # while the latter two always do
                        value = src if is_scalar else src[i, j, k]

                        if ((not include_self) and counts[tuple(ii)] == 0):
                            expected[tuple(ii)] = value
                        else:
                            if reduction == "add" or reduction == "sum":
                                expected[tuple(ii)] += value
                            elif reduction == "multiply" or reduction == "prod":
                                expected[tuple(ii)] *= value
                            elif reduction == "amax":
                                expected[tuple(ii)] = max(expected[tuple(ii)], value)
                            elif reduction == "amin":
                                expected[tuple(ii)] = min(expected[tuple(ii)], value)
                            elif reduction == "mean":
                                expected[tuple(ii)] += value
                            else:
                                expected[tuple(ii)] = value

                        counts[tuple(ii)] += 1

        if (reduction == "mean"):
            counts.masked_fill_(counts == 0, 1)
            if (dtype.is_floating_point or dtype.is_complex):
                expected /= counts
            else:
                expected.div_(counts, rounding_mode="floor")

        if dtype == torch.float16 or dtype == torch.bfloat16:
            # Some CUDA kernels (e.g. indexing_backward_kernel_stride_1) that are called during
            # the test use fp32 for internal accumulation for improved accuracy. When using 16 bit
            # precision types can be small differences
            self.assertEqual(actual, expected, atol=0.04, rtol=0.05)
        else:
            # When we are running opportunistic_fastatomics, we will expect some floating point rounding
            # errors as the order of operation is not guaranteed.
            if TEST_WITH_ROCM and CDNA3OrLater() \
                    and not torch.are_deterministic_algorithms_enabled():
                self.assertEqual(actual, expected, atol=1e-9, rtol=1e-6)
            else:
                self.assertEqual(actual, expected, atol=0, rtol=0)

        # Tests empty index
        dst = make_tensor((2, 2), device=device, dtype=dtype)
        idx = torch.tensor((), device=device, dtype=torch.long)
        src = make_tensor((2, 2), device=device, dtype=dtype)
        if reduction is not None:
            actual = fn(dst, 0, idx, src, reduce=reduction)
        else:
            actual = fn(dst, 0, idx, src)
        self.assertEqual(actual, dst, atol=0, rtol=0)

    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter_(self, device, dtype):
        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                        is_scalar=False, reduction=None)

    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter__scalar(self, device, dtype):
        self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                is_scalar=True, reduction=None)

    # FIXME: RuntimeError: "cuda_scatter_gather_base_kernel_reduce_multiply" not implemented for 'ComplexFloat'
    @toleranceOverride({torch.float16: tol(atol=1e-2, rtol=0)})
    @dtypesIfCUDA(torch.float16, torch.float32)
    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter__reductions(self, device, dtype):
        for reduction in ("add", "multiply"):
            self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                    is_scalar=False, reduction=reduction)
            self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                    is_scalar=True, reduction=reduction)

    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter_add_(self, device, dtype):
        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                self._test_scatter_base(torch.Tensor.scatter_add_, device=device, dtype=dtype,
                                        is_scalar=False, reduction=None)

    @dtypes(torch.float32)
    def test_scatter_add_mult_index_base(self, device, dtype):
        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                m, n = 30, 40
                idx = torch.zeros(m, n, device=device, dtype=torch.long)
                src = torch.ones(m, n, device=device, dtype=dtype)
                res0 = torch.zeros(m, n, device=device, dtype=dtype).scatter_add_(0, idx, src)
                res1 = torch.zeros(m, n, device=device, dtype=dtype).scatter_add_(1, idx, src)

                self.assertEqual(res0[0, :], m * torch.ones(n, device=device, dtype=dtype), atol=0, rtol=0)
                self.assertEqual(res1[:, 0], n * torch.ones(m, device=device, dtype=dtype), atol=0, rtol=0)

    # FIXME: discrepancy between bool ReduceAdd on CUDA and CPU (a + b on CPU and buggy a && b on CUDA)
    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=False))
    def test_scatter_reduce_sum(self, device, dtype):
        for include_self in (True, False):
            for deterministic in [False, True]:
                with DeterministicGuard(deterministic):
                    self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                            is_scalar=False, reduction='sum', unique_indices=False,
                                            include_self=include_self)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_prod(self, device, dtype):
        for include_self in (True, False):
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='prod', unique_indices=False,
                                    include_self=include_self)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_mean(self, device, dtype):
        for include_self in (True, False):
            for deterministic in [False, True]:
                with DeterministicGuard(deterministic):
                    self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                            is_scalar=False, reduction='mean', unique_indices=False,
                                            include_self=include_self)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_amax(self, device, dtype):
        for include_self in (True, False):
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='amax', unique_indices=False,
                                    include_self=include_self)
            # simple test for nan/inf propagation
            if (dtype.is_floating_point):
                input = torch.zeros(3, device=device, dtype=dtype)
                src = torch.tensor([1, float('nan'), -float('inf'), -float('inf'), 2, float('inf')], device=device, dtype=dtype)
                idx = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
                input.scatter_reduce_(0, idx, src, 'amax', include_self=include_self)
                expected_result = torch.tensor([float('nan'), -float('inf'), float('inf')], device=device, dtype=dtype)
                if (include_self):
                    expected_result[1] = 0
                self.assertEqual(input, expected_result)


    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_amin(self, device, dtype):
        for include_self in (True, False):
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='amin', unique_indices=False,
                                    include_self=include_self)
            # simple test for nan/inf propagation
            if (dtype.is_floating_point):
                input = torch.zeros(3, device=device, dtype=dtype)
                src = torch.tensor([1, float('nan'), -2, -float('inf'), float('inf'), float('inf')], device=device, dtype=dtype)
                idx = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
                input.scatter_reduce_(0, idx, src, 'amin', include_self=include_self)
                expected_result = torch.tensor([float('nan'), -float('inf'), float('inf')], device=device, dtype=dtype)
                if (include_self):
                    expected_result[2] = 0
                self.assertEqual(input, expected_result)

    @onlyCPU
    @dtypes(torch.float32, torch.float64, torch.bfloat16, torch.float16)
    def test_scatter_expanded_index(self, device, dtype):
        def helper(input_size, idx_size):
            input = torch.randn(input_size, device=device).to(dtype=dtype)
            input2 = input.clone()

            shape = [1] * len(input_size)
            shape[0] = idx_size
            dim_size = input_size[0]
            idx = torch.randint(0, dim_size, shape)

            # The fast path on scatter when index is expanded
            # will depend on sorted index where the collected src indice
            # for each row in input will be mapped to rowptrs in a CSR format.
            # Create some empty rows by masking:
            mask = (idx > 1) * (idx < 4)
            idx[mask] = 0

            expanded_shape = input_size
            expanded_shape[0] = idx_size
            idx = idx.expand(expanded_shape)
            idx2 = idx.contiguous()
            src = torch.randn(expanded_shape, device=device).to(dtype=dtype)

            out = input.scatter_add(0, idx, src)
            out2 = input2.scatter_add(0, idx2, src)
            self.assertEqual(out, out2)

            for reduce in ["sum", "prod", "mean", "amax", "amin"]:
                for include_self in [True, False]:
                    out = input.scatter_reduce(0, idx, src, reduce=reduce, include_self=include_self)
                    out2 = input2.scatter_reduce(0, idx2, src, reduce=reduce, include_self=include_self)
                    self.assertEqual(out, out2)

        helper([50, 17], 100)
        helper([50, 1], 100)
        helper([50, 8, 7], 100)
        helper([50, 3, 4, 5], 100)

    @dtypes(torch.float32)
    def test_scatter_add_broadcasted_index_deterministic(self, device, dtype):
        for d in (0, 1):
            inp = torch.randn(3, 4, 5, device=device, dtype=dtype)
            idx_1d = torch.randint(3, (10,), device=device)
            src_shape = list(inp.shape)
            src_shape[d] = 10
            src = torch.randn(src_shape, device=device, dtype=dtype)
            idx_view_shape = [1] * inp.ndim
            idx_view_shape[d] = 10
            idx = idx_1d.view(idx_view_shape).expand(src_shape)
            ref = inp.clone().scatter_add_(d, idx, src)
            with DeterministicGuard(True):
                res = inp.clone().scatter_add_(d, idx, src)
            self.assertEqual(res, ref)


    @onlyCPU
    @dtypes(torch.float32, torch.float64, torch.bfloat16)
    def test_gather_expanded_index(self, device, dtype):
        # Test when index is [N, 1], which would have stride [1, 0]
        # should be excluded from the fast path when index ix expanded
        input = torch.arange(25).view(5, 5)
        input2 = input.to(dtype=dtype)

        idx = torch.arange(5).view(5, 1)
        out = torch.gather(input, 0, idx)
        out2 = torch.gather(input2, 0, idx)

        self.assertEqual(out.to(dtype=dtype), out2)

        def helper(input_size, idx_size):
            input = torch.randn(input_size, device=device).to(dtype=dtype)
            input2 = input.clone()

            shape = [1] * len(input_size)
            shape[0] = idx_size
            dim_size = input_size[0]
            idx = torch.randint(0, dim_size, shape)

            # Test the fast path on gather when index is expanded
            expanded_shape = input_size
            expanded_shape[0] = idx_size
            idx = idx.expand(expanded_shape)
            idx2 = idx.contiguous()

            out = torch.gather(input, 0, idx)
            out2 = torch.gather(input2, 0, idx2)

            self.assertEqual(out, out2)

            # test unsqueezed index
            # expanded_index kernel can not handle the case:
            # the size > 1 and stride == 1 at a dimension.
            # for example: the index with size of [1, 8, 7],  stride of [1, 1, 0].
            # see https://github.com/pytorch/pytorch/issues/129093
            def unsqueeze_helper(idx, dim):
                if dim == 2:
                    return idx.unsqueeze(1).t()
                else:
                    return unsqueeze_helper(idx, dim - 1).unsqueeze(dim - 1)

            idx = torch.randint(0, dim_size, (input.shape[1],))
            idx = unsqueeze_helper(idx, len(input_size))
            expanded_shape[0] = 1
            idx = idx.expand(expanded_shape)
            idx2 = idx.contiguous()
            out = torch.gather(input, 0, idx)
            out2 = torch.gather(input2, 0, idx2)
            self.assertEqual(out, out2)

        helper([50, 17], 100)
        helper([50, 1], 100)
        helper([50, 8, 7], 100)
        helper([50, 3, 4, 5], 100)

# Generic Device Test Framework instantiation, see
#   https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
#   for details.
instantiate_device_type_tests(TestScatterGather, globals())

if __name__ == '__main__':
    run_tests()
