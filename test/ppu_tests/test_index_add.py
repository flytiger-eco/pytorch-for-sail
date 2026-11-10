# Owner(s): ["module: tests"]
# PPU paired-atomic index_add coverage, including native fallback sentinels.
# Run: python -m pytest test/ppu_tests/test_index_add.py

import operator
from functools import reduce

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyOn,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    parametrize,
    run_tests,
    serialTest,
    TestCase,
)


class TestPPUIndexAdd(TestCase):
    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("shape", [(), (7,), (7, 6), (3, 7, 5), (2, 3, 7, 4),
                           (2, 1, 3, 7, 3), (1,) * 24])
    @parametrize("index_layout", ["dense", "strided", "broadcast"])
    @parametrize("offsets", [(0, 0), (1, 0), (0, 1), (1, 1)])
    @parametrize("alpha", [0, 1, -0.5, 2])
    def test_index_add_pairs_nd(self, device, dtype, shape, index_layout, offsets, alpha):
        rank = len(shape)
        dims = (0, rank // 2, rank - 1) if rank > 5 else range(max(1, rank))
        for d in dims:
            output_shape, source_shape = list(shape), list(shape)
            if rank:
                output_shape[d], source_shape[d] = 7, 11
            source_numel = reduce(operator.mul, source_shape, 1)
            output_numel = reduce(operator.mul, output_shape, 1)
            source_cpu = ((torch.arange(source_numel).reshape(source_shape) % 7 - 3).float() * 0.125).to(dtype)
            initial = torch.full(output_shape, 0.125, dtype=dtype)
            index_cpu = (torch.arange(11) * 3) % 7 if rank else torch.tensor(0)
            if index_layout == "broadcast":
                index_cpu = torch.zeros_like(index_cpu)
            scaled = (source_cpu.float() * torch.tensor(alpha, dtype=dtype).float()).to(dtype)
            expected = initial.float().index_add(d, index_cpu, scaled.float()).to(dtype)
            for dim in (d, d - max(1, rank)):
                output_offset, source_offset = offsets
                source_storage = torch.full((source_numel + 4,), 17.0, device=device, dtype=dtype)
                source = source_storage[source_offset:source_offset + source_numel].reshape(source_shape)
                source.copy_(source_cpu)
                preserved_source = source_storage.clone()
                output_storage = torch.full((output_numel + 4,), 9.0, device=device, dtype=dtype)
                output = output_storage[output_offset:output_offset + output_numel].reshape(output_shape)
                output.copy_(initial)
                index = index_cpu.to(device)
                if rank and index_layout == "strided":
                    index = index.repeat_interleave(2)[::2]
                elif rank and index_layout == "broadcast":
                    index = torch.zeros(1, device=device, dtype=torch.long).expand(11)
                with DeterministicGuard(False):
                    output.index_add_(dim, index, source, alpha=alpha)
                    functional = torch.index_add(initial.to(device), dim, index, source, alpha=alpha)
                    out = torch.empty_like(output)
                    torch.index_add(initial.to(device), dim, index, source, alpha=alpha, out=out)
                bits = torch.int32 if dtype == torch.float32 else torch.int16
                for actual in (output, functional, out):
                    self.assertEqual(actual.cpu().view(bits), expected.view(bits))
                self.assertEqual(source_storage, preserved_source)
                self.assertEqual(output_storage[:output_offset], torch.full_like(output_storage[:output_offset], 9.0))
                tail = output_storage[output_offset + output_numel:]
                self.assertEqual(tail, torch.full_like(tail, 9.0))

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("dim", [0, 1, 2, -1])
    @parametrize("alpha", [0.1, -0.3, 1.0006, 3.141592653589793])
    def test_index_add_pair_alpha_rounding_nd(self, device, dtype, dim, alpha):
        shape = [3, 7, 5]
        shape[dim] = 7
        numel = reduce(operator.mul, shape, 1)
        source_cpu = ((torch.arange(numel).reshape(shape) % 127 - 63).float() / 13).to(dtype)
        index_cpu = (torch.arange(7) * 3) % 7
        scaled = (source_cpu.float() * torch.tensor(alpha, dtype=dtype).float()).to(dtype)
        expected = torch.zeros(shape).index_add(dim, index_cpu, scaled.float()).to(dtype)
        source_storage = torch.full((numel + 2,), 17.0, device=device, dtype=dtype)
        source = source_storage[1:numel + 1].reshape(shape).copy_(source_cpu)
        output_storage = torch.full((numel + 2,), 9.0, device=device, dtype=dtype)
        output = output_storage[1:numel + 1].reshape(shape).zero_()
        with DeterministicGuard(False):
            output.index_add_(dim, index_cpu.to(device), source, alpha=alpha)
        self.assertEqual(output.cpu().view(torch.int16), expected.view(torch.int16))
        self.assertEqual(output_storage[[0, -1]], torch.full((2,), 9.0, device=device, dtype=dtype))
        self.assertEqual(source.cpu().view(torch.int16), source_cpu.view(torch.int16))

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("shape, dim", [((7,), 0), ((3, 7, 5), 1), ((2, 3, 7, 4), 2),
                                ((3, 7, 5), -1), ((1,) * 12 + (7,) + (1,) * 11, 12)])
    @parametrize("deterministic", [False, True])
    def test_index_select_pair_backward_nd(self, device, dtype, shape, dim, deterministic):
        d = dim % len(shape)
        grad_shape = list(shape)
        grad_shape[d] = 11
        index_cpu = (torch.arange(11) * 3) % shape[d]
        numel = reduce(operator.mul, grad_shape, 1)
        grad_cpu = ((torch.arange(numel).reshape(grad_shape) % 7 - 3).float() * 0.125).to(dtype)
        expected = torch.zeros(shape).index_add(dim, index_cpu, grad_cpu.float()).to(dtype)
        leaf = torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)
        with DeterministicGuard(deterministic):
            leaf.index_select(dim, index_cpu.to(device)).backward(grad_cpu.to(device))
        self.assertEqual(leaf.grad.cpu().view(torch.int16), expected.view(torch.int16))

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("deterministic", [False, True])
    @parametrize("offset", [0, 1])
    def test_index_add_pair_nd_dispatch(self, device, dtype, deterministic, offset):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        initial = torch.zeros((3, 7, 5), device=device, dtype=dtype)
        source = torch.full((3, 11, 5), 0.125, device=device, dtype=dtype)
        storage = torch.zeros(initial.numel() + 2, device=device, dtype=dtype)
        output = storage[offset:offset + initial.numel()].reshape(initial.shape)
        index = (torch.arange(11, device=device) % 7).repeat_interleave(2)[::2]
        with DeterministicGuard(deterministic), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            output.index_add_(-2, index, source)
        expected = torch.zeros((3, 7, 5)).index_add(1, index.cpu(), source.float().cpu()).to(dtype)
        self.assertEqual(output.cpu().view(torch.int16), expected.view(torch.int16))
        kernels = [event.name for event in prof.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        hit = any("ppu_index_scatter_opt::pair_atomic_add_" in name for name in kernels)
        self.assertFalse(hit)
        if not deterministic:
            self.assertRegex(" ".join(kernels), r"indexFunc(?:Small|Large)Index")

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @serialTest()
    @largeTensorTest("5GB")
    def test_index_add_pairs_nd_64bit_addressing(self, device, dtype):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        size = (1 << 31) + 5
        output = torch.zeros((1, size, 1), device=device, dtype=dtype)
        source = torch.full((1, 7, 1), 0.125, device=device, dtype=dtype)
        index = torch.tensor([size - 1, size - 2, 0, 1, 2, 3, 4], device=device)
        with DeterministicGuard(False), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            output.index_add_(-2, index, source)
        self.assertEqual(output.count_nonzero().item(), 7)
        self.assertEqual(output[0, index, 0], torch.full((7,), 0.125, device=device, dtype=dtype))
        kernels = [event.name for event in prof.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        self.assertRegex(" ".join(kernels),
                         r"indexFunc(?:Small|Large)Index<[^,]+,\s*[^,]+,\s*(?:unsigned long(?: long)?|uint64_t)\s*,")

    def _make_index_add_pair_layout(self, values, device, layout, dim, offset):
        shape = list(values.shape)
        base_shape = list(shape)
        if layout == "broadcast" and shape:
            base_shape[dim] = 1
        numel = reduce(operator.mul, base_shape, 1)
        step = 2 if layout == "strided" else 1
        storage = torch.full((numel * step + 4,), -0.0, device=device, dtype=values.dtype)
        flat = storage[offset:offset + numel * step:step]
        if layout == "transpose" and len(shape) > 1:
            view = flat.reshape(list(reversed(shape))).permute(list(reversed(range(len(shape)))))
        else:
            view = flat.reshape(base_shape)
        if layout == "broadcast" and shape:
            view.copy_(values.select(dim, 0).unsqueeze(dim))
            view = view.expand(shape)
        else:
            view.copy_(values)
        locations = torch.arange(storage.numel()).as_strided(view.shape, view.stride(), view.storage_offset())
        return view, storage, locations

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("shape", [(), (7,), (7, 6), (3, 7, 5), (2, 3, 7, 4), (1,) * 24])
    @parametrize("layouts", [("contiguous", "contiguous"), ("transpose", "contiguous"),
                             ("contiguous", "strided"), ("strided", "transpose"),
                             ("strided", "broadcast"), ("transpose", "strided"),
                             ("contiguous", "broadcast"), ("strided", "contiguous")])
    @parametrize("index_layout", ["dense", "strided", "broadcast"])
    @parametrize("alpha", [0, 1, -0.5, 2])
    @parametrize("offset", [0, 1])
    def test_index_add_pairs_layouts(self, device, dtype, index_dtype, shape, layouts, index_layout, alpha, offset):
        rank = len(shape)
        dims = (0, rank // 2, rank - 1) if rank > 4 else range(max(1, rank))
        for d in dims:
            output_shape, source_shape = list(shape), list(shape)
            if rank:
                output_shape[d], source_shape[d] = 7, 11
            numel = reduce(operator.mul, source_shape, 1)
            source_cpu = ((torch.arange(numel).reshape(source_shape) % 7 - 3).float() * 0.125).to(dtype)
            if layouts[1] == "broadcast":
                source_cpu.fill_(0.125)
            initial = torch.full(output_shape, 0.125, dtype=dtype)
            index_cpu = ((torch.arange(11) * 3) % 7).to(index_dtype) if rank else torch.tensor(0, dtype=index_dtype)
            if index_layout == "broadcast":
                index_cpu.zero_()
            scaled = (source_cpu.float() * torch.tensor(alpha, dtype=dtype).float()).to(dtype)
            expected = initial.float().index_add(d, index_cpu, scaled.float()).to(dtype)
            for dim in (d, d - max(1, rank)):
                source, source_storage, _ = self._make_index_add_pair_layout(source_cpu, device, layouts[1], d, offset)
                output, output_storage, locations = self._make_index_add_pair_layout(initial, device, layouts[0], d, offset)
                out, out_storage, _ = self._make_index_add_pair_layout(initial, device, layouts[0], d, offset)
                preserved_source = source_storage.cpu().clone()
                expected_storage = output_storage.cpu().clone()
                expected_storage[locations.reshape(-1)] = expected.reshape(-1)
                index = index_cpu.to(device)
                if rank and index_layout == "strided":
                    index = index.repeat_interleave(2)[::2]
                elif rank and index_layout == "broadcast":
                    index = torch.zeros(1, device=device, dtype=index_dtype).expand(11)
                with DeterministicGuard(False):
                    output.index_add_(dim, index, source, alpha=alpha)
                    functional = torch.index_add(initial.to(device), dim, index, source, alpha=alpha)
                    torch.index_add(initial.to(device), dim, index, source, alpha=alpha, out=out)
                bits = torch.int32 if dtype == torch.float32 else torch.int16
                for actual in (output, functional, out):
                    self.assertEqual(actual.cpu().view(bits), expected.view(bits))
                untouched = torch.ones_like(expected_storage, dtype=torch.bool)
                untouched[locations.reshape(-1)] = False
                zero_guards = untouched & (expected_storage == 0)
                for backing in (output_storage, out_storage):
                    actual_storage = backing.cpu()
                    # Native single-value atomics may change an untouched zero's sign, never its value.
                    self.assertEqual(actual_storage[zero_guards], expected_storage[zero_guards], atol=0, rtol=0)
                    actual_storage[zero_guards] = expected_storage[zero_guards]
                    self.assertEqual(actual_storage.view(bits), expected_storage.view(bits))
                self.assertEqual(source_storage.cpu().view(bits), preserved_source.view(bits))

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("dim", [0, 1, 2, -1])
    @parametrize("alpha", [0.1, -0.3, 1.0006, 3.141592653589793])
    @parametrize("offset", [0, 1])
    def test_index_add_pairs_layout_alpha(self, device, dtype, index_dtype, dim, alpha, offset):
        shape = [3, 7, 5]
        shape[dim] = 7
        numel = reduce(operator.mul, shape, 1)
        values = ((torch.arange(numel).reshape(shape) % 127 - 63).float() / 13).to(dtype)
        index = ((torch.arange(7) * 3) % 7).to(index_dtype)
        scaled = (values.float() * torch.tensor(alpha, dtype=dtype).float()).to(dtype)
        expected = torch.zeros(shape).index_add(dim, index, scaled.float()).to(dtype)
        for output_layout, source_layout in [("transpose", "strided"), ("strided", "transpose"),
                                             ("contiguous", "strided")]:
            source, source_storage, _ = self._make_index_add_pair_layout(values, device, source_layout, dim, offset)
            output, storage, locations = self._make_index_add_pair_layout(torch.zeros_like(values), device, output_layout, dim, offset)
            preserved_source = source_storage.cpu().clone()
            expected_storage = storage.cpu().clone()
            expected_storage[locations.reshape(-1)] = expected.reshape(-1)
            with DeterministicGuard(False):
                output.index_add_(dim, index.to(device), source, alpha=alpha)
            actual_storage = storage.cpu()
            untouched = torch.ones_like(expected_storage, dtype=torch.bool)
            untouched[locations.reshape(-1)] = False
            zero_guards = untouched & (expected_storage == 0)
            # Normalize only untouched zero signs; logical outputs and nonzero guards remain bitwise checked.
            self.assertEqual(actual_storage[zero_guards], expected_storage[zero_guards], atol=0, rtol=0)
            actual_storage[zero_guards] = expected_storage[zero_guards]
            self.assertEqual(actual_storage.view(torch.int16), expected_storage.view(torch.int16))
            self.assertEqual(source_storage.cpu().view(torch.int16), preserved_source.view(torch.int16))

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("shape, dim", [((7,), 0), ((3, 7, 5), 1), ((2, 3, 7, 4), 2),
                                ((3, 7, 5), -1), ((1,) * 12 + (7,) + (1,) * 11, 12)])
    @parametrize("layout", ["transpose", "strided", "broadcast"])
    @parametrize("deterministic", [False, True])
    def test_index_select_pairs_layout_backward(self, device, dtype, index_dtype, shape, dim, layout, deterministic):
        d = dim % len(shape)
        grad_shape = list(shape)
        grad_shape[d] = 11
        numel = reduce(operator.mul, grad_shape, 1)
        grad_cpu = ((torch.arange(numel).reshape(grad_shape) % 7 - 3).float() * 0.125).to(dtype)
        if layout == "broadcast":
            grad_cpu.fill_(0.125)
        index = ((torch.arange(11) * 3) % shape[d]).to(index_dtype)
        grad, _, _ = self._make_index_add_pair_layout(grad_cpu, device, layout, d, 1)
        leaf, _, _ = self._make_index_add_pair_layout(torch.zeros(shape, dtype=dtype), device, "transpose", d, 1)
        leaf = leaf.detach().requires_grad_(True)
        expected = torch.zeros(shape).index_add(dim, index, grad_cpu.float()).to(dtype)
        with DeterministicGuard(deterministic):
            leaf.index_select(dim, index.to(device)).backward(grad)
        self.assertEqual(leaf.grad.cpu().view(torch.int16), expected.view(torch.int16))

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("layouts", [("contiguous", "contiguous"), ("transpose", "strided"),
                             ("strided", "transpose"), ("contiguous", "broadcast")])
    @parametrize("dim", [0, 1, -1])
    @parametrize("alpha", [1, -0.5])
    @parametrize("offset", [0, 1])
    @parametrize("deterministic", [False, True])
    def test_index_add_pairs_layout_dispatch(self, device, dtype, index_dtype, layouts, dim, alpha, offset, deterministic):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        output_shape, source_shape = [3, 7, 6], [3, 7, 6]
        output_shape[dim], source_shape[dim] = 7, 11
        initial = torch.zeros(output_shape, dtype=dtype)
        values = torch.full(source_shape, 0.125, dtype=dtype)
        source, _, _ = self._make_index_add_pair_layout(values, device, layouts[1], dim, offset)
        output, _, _ = self._make_index_add_pair_layout(initial, device, layouts[0], dim, offset)
        index = (torch.arange(11, device=device, dtype=index_dtype) % 7).repeat_interleave(2)[::2]
        with DeterministicGuard(deterministic), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            output.index_add_(dim, index, source, alpha=alpha)
        expected = initial.float().index_add(dim, index.cpu(), values.float(), alpha=alpha).to(dtype)
        self.assertEqual(output.cpu().view(torch.int32 if dtype == torch.float32 else torch.int16),
                         expected.view(torch.int32 if dtype == torch.float32 else torch.int16))
        kernels = [event.name for event in prof.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        paired = [name for name in kernels if "ppu_index_scatter_opt::pair_atomic_add_" in name]
        expected_hit = (dtype != torch.float32 and not deterministic and dim == 0 and
                        layouts == ("contiguous", "contiguous") and offset == 0)
        self.assertEqual(bool(paired), expected_hit)
        if not paired and not deterministic:
            self.assertRegex(" ".join(kernels), r"indexFunc(?:Small|Large)Index")
        if paired and index_dtype == torch.int32:
            self.assertRegex(" ".join(paired), r"pair_atomic_add_(?:nd_)?kernel<[^,]+,\s*[^,]+,\s*(?:int|int32_t)\s*,")

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("large_tensor", ["output", "source"])
    @serialTest()
    @largeTensorTest("5GB")
    def test_index_add_pairs_layout_64bit(self, device, dtype, index_dtype, large_tensor):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        span = (1 << 31) + 1
        storage = torch.full((span + 8,), -0.0, device=device, dtype=dtype)
        source = torch.full((2, 5, 1), 0.125, device=device, dtype=dtype)
        output = torch.zeros((2, 3, 1), device=device, dtype=dtype)
        if large_tensor == "output":
            output = storage.as_strided((2, 3, 1), (span, 1, 1), 1).zero_()
        else:
            source = storage.as_strided((2, 5, 1), (span, 1, 1), 1).copy_(source)
        index = torch.tensor([0, 2, 1, 0, 2], device=device, dtype=index_dtype)
        gap_offsets = [0, 6, span, span + 6, span + 7]
        if large_tensor == "output":
            gap_offsets.extend([4, span + 4])
        gaps = torch.tensor(gap_offsets, device=device)
        storage[gaps] = 9.0
        preserved = storage[gaps].clone()
        with DeterministicGuard(False), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            output.index_add_(-2, index, source)
        expected = torch.zeros((2, 3, 1)).index_add(1, index.cpu(), torch.full((2, 5, 1), 0.125)).to(dtype)
        self.assertEqual(output.cpu().view(torch.int16), expected.view(torch.int16))
        self.assertEqual(source, torch.full((2, 5, 1), 0.125, device=device, dtype=dtype))
        self.assertEqual(storage[gaps].view(torch.int16), preserved.view(torch.int16))
        self.assertEqual(storage.count_nonzero().item(),
                         (6 if large_tensor == "output" else 10) + len(gap_offsets))
        kernels = [event.name for event in prof.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        self.assertRegex(" ".join(kernels),
                         r"indexFunc(?:Small|Large)Index<[^,]+,\s*[^,]+,\s*(?:unsigned long(?: long)?|uint64_t)\s*,")

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("layout", ["contiguous", "strided", "broadcast"])
    def test_index_add_pair_rank_compatibility_dispatch(self, device, dtype, index_dtype, layout):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        shape = (1,) * 12 + (7,) + (1,) * 12
        source_shape = (1,) * 12 + (11,) + (1,) * 12
        values = torch.full(source_shape, 0.125, dtype=dtype)
        source, preserved, _ = self._make_index_add_pair_layout(values, device, layout, 12, 1)
        source_before = preserved.clone()
        output = torch.zeros(shape, device=device, dtype=dtype)
        index = (torch.arange(11, device=device, dtype=index_dtype) * 3) % 7
        with DeterministicGuard(False), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            output.index_add_(-13, index, source)
            torch.cuda.synchronize()
        expected = torch.zeros(7).index_add(0, index.long().cpu(), torch.full((11,), 0.125)).to(dtype)
        self.assertEqual(output.cpu().reshape(7).view(torch.int16), expected.view(torch.int16))
        self.assertEqual(preserved.view(torch.int16), source_before.view(torch.int16))
        kernels = [event.name for event in prof.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        self.assertTrue(any("ppu_index_scatter_opt::pair_atomic_add_nd_kernel" in name for name in kernels))

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    def test_index_add_pairs_layout_overlap(self, device, dtype, index_dtype):
        index = torch.arange(7, device=device, dtype=index_dtype)
        source = torch.full((3, 7, 5), 0.125, device=device, dtype=dtype)
        expanded = torch.zeros((1, 7, 5), device=device, dtype=dtype).expand(3, 7, 5)
        aliased = torch.zeros((5, 7, 3), device=device, dtype=dtype).permute(2, 1, 0)
        with DeterministicGuard(False):
            with self.assertRaises(RuntimeError):
                expanded.index_add_(1, index, source)
            with self.assertRaises(RuntimeError):
                aliased.index_add_(1, index, aliased)


instantiate_device_type_tests(TestPPUIndexAdd, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
