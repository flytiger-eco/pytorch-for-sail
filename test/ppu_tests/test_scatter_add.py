# Owner(s): ["module: scatter & gather ops"]
# PPU paired-atomic scatter_add coverage, including native fallback sentinels.
# Run: python -m pytest test/ppu_tests/test_scatter_add.py

import operator
from functools import reduce

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    parametrize,
    run_tests,
    serialTest,
    TestCase,
)


class TestPPUScatterAdd(TestCase):
    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("shape", [(), (7,), (7, 6), (3, 7, 5), (2, 3, 7, 4),
                           (2, 1, 3, 7, 3), (1,) * 24])
    @parametrize("index_layout", ["broadcast", "dense", "strided", "permuted"])
    @parametrize("partial", [False, True])
    @parametrize("offsets", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_scatter_add_pairs_nd(self, device, dtype, index_dtype, shape, index_layout, partial, offsets):
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
                ids = (torch.arange(index_shape[d], dtype=index_dtype) * 3) % 7
                index_cpu = ids.reshape(index_view).expand(index_shape)
            else:
                index_cpu = ((torch.arange(index_numel, dtype=index_dtype) * 3) % 7).reshape(index_shape)
            expected = initial.float().scatter_add(d, index_cpu.long(), source_cpu.float()).to(dtype)
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
                index_before = index.cpu().clone()
                self.assertEqual(index.dtype, index_dtype)
                with DeterministicGuard(False):
                    output.scatter_add_(dim, index, source)
                    functional = torch.scatter_add(initial.to(device), dim, index, source)
                    out = torch.empty_like(output)
                    torch.scatter_add(initial.to(device), dim, index, source, out=out)
                bits = torch.int32 if dtype == torch.float32 else torch.int16
                for actual in (output, functional, out):
                    self.assertEqual(actual.cpu().view(bits), expected.view(bits))
                self.assertEqual(source_storage, preserved_source)
                self.assertEqual(index.cpu(), index_before)
                self.assertEqual(output_storage[:output_offset], torch.full_like(output_storage[:output_offset], 9.0))
                tail = output_storage[output_offset + output_numel:]
                self.assertEqual(tail, torch.full_like(tail, 9.0))

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("shape, dim", [((7,), 0), ((7, 8), 0), ((7, 3, 8), 0),
                                ((3, 7, 5), 1), ((2, 3, 7, 4), 2),
                                ((3, 7, 5), -1), ((1,) * 12 + (7,) + (1,) * 11, 12)])
    @parametrize("deterministic", [False, True])
    def test_gather_pair_backward_nd(self, device, dtype, index_dtype, shape, dim, deterministic):
        d = dim % len(shape)
        grad_shape, index_view = list(shape), [1] * len(shape)
        grad_shape[d], index_view[d] = 11, 11
        ids = (torch.arange(11, dtype=index_dtype) * 3) % shape[d]
        index_cpu = ids.reshape(index_view).expand(grad_shape)
        numel = reduce(operator.mul, grad_shape, 1)
        grad_cpu = ((torch.arange(numel).reshape(grad_shape) % 7 - 3).float() * 0.125).to(dtype)
        expected = torch.zeros(shape).scatter_add(dim, index_cpu.long(), grad_cpu.float()).to(dtype)
        leaf = torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)
        index = ids.to(device).reshape(index_view).expand(grad_shape)
        with DeterministicGuard(deterministic):
            leaf.gather(dim, index).backward(grad_cpu.to(device))
        self.assertEqual(leaf.grad.cpu().view(torch.int16), expected.view(torch.int16))

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("index_layout", ["broadcast", "dense", "strided", "partial"])
    @parametrize("deterministic", [False, True])
    @parametrize("offset", [0, 1])
    def test_scatter_add_pair_nd_dispatch(self, device, dtype, index_dtype, index_layout, deterministic, offset):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        shape = (3, 7, 5)
        source_shape = (5, 11, 7) if index_layout == "partial" else (3, 11, 5)
        index_shape = (2, 9, 4) if index_layout == "partial" else source_shape
        source = torch.full(source_shape, 0.125, device=device, dtype=dtype)
        storage = torch.zeros(3 * 7 * 5 + 2, device=device, dtype=dtype)
        output = storage[offset:offset + 3 * 7 * 5].reshape(shape)
        if index_layout == "broadcast":
            index = (torch.arange(11, device=device, dtype=index_dtype) % 7).reshape(1, 11, 1).expand(index_shape)
        else:
            numel = reduce(operator.mul, index_shape, 1)
            index = (torch.arange(numel, device=device, dtype=index_dtype) % 7).reshape(index_shape)
            if index_layout == "strided":
                index = index.repeat_interleave(2, dim=-1)[..., ::2]
        with DeterministicGuard(deterministic), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            output.scatter_add_(-2, index, source)
        expected = torch.zeros(shape).scatter_add(1, index.long().cpu(), source.float().cpu()).to(dtype)
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
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("shape, dim", [((11, 8), 0), ((11, 3, 8), 0), ((3, 11, 8), 1), ((11, 7), 0)])
    @parametrize("index_layout", ["broadcast", "dense_same", "dense_mixed", "strided", "partial"])
    @parametrize("offset", [0, 1])
    @parametrize("deterministic", [False, True])
    def test_scatter_add_pair_fastpath_dispatch(self, device, dtype, index_dtype, shape, dim, index_layout, offset, deterministic):
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
        index = (torch.arange(shape[dim], device=device, dtype=index_dtype) * 3 % 7).reshape(ids_shape).expand(shape)
        if index_layout != "broadcast":
            index = index.contiguous()
        if index_layout == "dense_mixed":
            index = (index + torch.arange(index.numel(), device=device, dtype=index_dtype).reshape(shape)) % 7
        elif index_layout == "strided":
            index = index.repeat_interleave(2, dim=-1)[..., ::2]
        elif index_layout == "partial":
            index = index.narrow(dim, 0, 9)
        expected = torch.zeros(output_shape).scatter_add(dim, index.long().cpu(), source.float().cpu()).to(dtype)
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
        paired = [name for name in kernels if "ppu_index_scatter_opt::pair_atomic_add_kernel" in name]
        hit = bool(paired)
        inner = reduce(operator.mul, shape[dim + 1:], 1)
        expected_hit = (dtype != torch.float32 and not deterministic and dim == 0 and inner % 2 == 0 and
                        offset == 0 and index_layout in ("broadcast", "dense_same", "dense_mixed"))
        self.assertEqual(hit, expected_hit)
        if paired:
            index_type = r"(?:int|int32_t)" if index_dtype == torch.int32 else r"(?:long(?: long)?|int64_t)"
            self.assertRegex(" ".join(paired), rf"pair_atomic_add_kernel<[^,]+,\s*[^,]+,\s*{index_type}\s*,")
        if not expected_hit and not deterministic:
            self.assertTrue(any("scatter_gather" in name for name in kernels))

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("index_layout", ["broadcast", "dense_same", "dense_mixed", "strided"])
    @parametrize("dim", [0, -2])
    def test_gather_pair_fastpath_backward_dispatch(self, device, dtype, index_dtype, index_layout, dim):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        leaf = torch.zeros((7, 8), device=device, dtype=dtype, requires_grad=True)
        grad = torch.full((11, 8), 0.125, device=device, dtype=dtype)
        index = (torch.arange(11, device=device, dtype=index_dtype) * 3 % 7).reshape(11, 1).expand_as(grad)
        if index_layout != "broadcast":
            index = index.contiguous()
        if index_layout == "dense_mixed":
            index = (index + torch.arange(index.numel(), device=device, dtype=index_dtype).reshape(grad.shape)) % 7
        elif index_layout == "strided":
            index = index.repeat_interleave(2, dim=-1)[..., ::2]
        expected = torch.zeros((7, 8)).scatter_add(dim, index.long().cpu(), grad.float().cpu()).to(dtype)
        with DeterministicGuard(False), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            leaf.gather(dim, index).backward(grad)
            torch.cuda.synchronize()
        self.assertEqual(leaf.grad.cpu().view(torch.int16), expected.view(torch.int16))
        kernels = [event.name for event in prof.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        paired = [name for name in kernels if "ppu_index_scatter_opt::pair_atomic_add_kernel" in name]
        self.assertEqual(bool(paired), index_layout != "strided")
        if paired:
            index_type = r"(?:int|int32_t)" if index_dtype == torch.int32 else r"(?:long(?: long)?|int64_t)"
            self.assertRegex(" ".join(paired), rf"pair_atomic_add_kernel<[^,]+,\s*[^,]+,\s*{index_type}\s*,")
        else:
            self.assertTrue(any("scatter_gather" in name for name in kernels))

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("index_layout", ["broadcast", "dense_same", "dense_mixed"])
    @serialTest()
    @largeTensorTest("5GB")
    def test_scatter_add_pair_fastpath_64bit_addressing(self, device, dtype, index_dtype, index_layout):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        rows = (1 << 28) + 1
        output = torch.zeros((rows, 8), device=device, dtype=dtype)
        source = torch.full((2, 8), 0.125, device=device, dtype=dtype)
        index = torch.tensor([0, rows - 1], device=device, dtype=index_dtype).reshape(2, 1).expand_as(source)
        if index_layout != "broadcast":
            index = index.contiguous()
        if index_layout == "dense_mixed":
            index = (torch.arange(16, device=device, dtype=index_dtype) % 2 * (rows - 1)).reshape(2, 8)
        local_index = (index.cpu() == rows - 1).to(index_dtype)
        expected = torch.zeros((2, 8)).scatter_add(0, local_index.long(), source.float().cpu()).to(dtype)
        source_before = source.clone()
        self.assertGreater(output.numel(), torch.iinfo(torch.int32).max)
        with DeterministicGuard(False), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            output.scatter_add_(0, index, source)
            torch.cuda.synchronize()
        self.assertEqual(output[[0, rows - 1]].cpu().view(torch.int16), expected.view(torch.int16))
        self.assertEqual(output.count_nonzero().item(), expected.count_nonzero().item())
        self.assertEqual(source.view(torch.int16), source_before.view(torch.int16))
        kernels = [event.name for event in prof.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        paired = [name for name in kernels if "ppu_index_scatter_opt::pair_atomic_add_kernel" in name]
        self.assertTrue(paired)
        index_type = r"(?:int|int32_t)" if index_dtype == torch.int32 else r"(?:long(?: long)?|int64_t)"
        self.assertRegex(" ".join(paired),
                         rf"pair_atomic_add_kernel<[^,]+,\s*(?:unsigned long(?: long)?|uint64_t),\s*{index_type}\s*,")


instantiate_device_type_tests(TestPPUScatterAdd, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
