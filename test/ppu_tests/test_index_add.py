

# PPU paired-atomic index_add coverage, including native fallback sentinels.
# Run: python -m pytest test/ppu_tests/test_index_add.py

import operator
import os
import subprocess
import sys
import textwrap
import unittest
from contextlib import nullcontext
from functools import reduce, wraps

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


_INDEX_ADD_OPT_ENABLED = os.getenv("PYTORCH_DISABLE_PPU_INDEX_ADD_OPT") != "1"


def _count_nonzero_chunked(tensor, chunk_size=1 << 20):
    # Keep bool/int64 reduction temporaries bounded without copying a large view.
    if not tensor.is_contiguous():
        raise ValueError("Chunked counting requires a contiguous tensor")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    flat = tensor.view(-1)
    return sum(flat[start:start + chunk_size].count_nonzero().item()
               for start in range(0, flat.numel(), chunk_size))


def _representative_matrix(fn):
    if os.getenv("PPU_ATOMIC_FULL_MATRIX") == "1":
        return fn
    full_parametrize = fn.parametrize_fn

    def representative(test, generic_cls, device_cls):
        cases = list(full_parametrize(test, generic_cls, device_cls))
        anchor = cases[0][2]
        # Keep every one- and two-factor interaction; repeat them for both index dtypes.
        for case in cases:
            differences = sum(value != anchor[name] for name, value in case[2].items()
                              if name != "index_dtype")
            if differences <= 2:
                yield case

    fn.parametrize_fn = representative
    return fn


_TRACING_DEVICES = set()


def _device_kernels(prof):
    if prof is None:
        return None
    return [event.name for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA]


def _require_kernel_tracing(device):
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise unittest.SkipTest("CUDA kernel tracing is unavailable; numerical cases run separately")
    key = str(device)
    if key not in _TRACING_DEVICES:
        # Probe an unrelated kernel before either operator's first-entry cache is initialized.
        probe = torch.ones(8, device=device)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            probe.add_(1)
            torch.cuda.synchronize(device)
        if not _device_kernels(prof):
            raise AssertionError("CUDA tracing was advertised but the device-kernel probe captured no events")
        _TRACING_DEVICES.add(key)


def _with_route_check(fn):
    @wraps(fn)
    def wrapped(self, device, *args, check_route=False, **kwargs):
        self._check_route = check_route
        if check_route:
            _require_kernel_tracing(device)
        return fn(self, device, *args, **kwargs)
    return parametrize("check_route", [False, True])(wrapped)


_SUBPROCESS_TRACING = textwrap.dedent(r"""
    import os
    import sys
    import torch
    from contextlib import nullcontext
    check_route = os.environ.get("PPU_ATOMIC_ROUTE_CHECK") == "1"
    def capture():
        if not check_route:
            return nullcontext()
        return torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ])
    def device_kernels(prof):
        if prof is None:
            return None
        return [event.name for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA]
    if check_route:
        assert torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()
        device = sys.argv[1]
        probe = torch.ones(8, device=device)
        torch.cuda.synchronize(device)
        with capture() as prof:
            probe.add_(1)
            torch.cuda.synchronize(device)
        assert device_kernels(prof), "Subprocess device-kernel tracing probe failed"
""")


class TestPPUIndexAdd(TestCase):
    def _capture_pair_profile(self, device):
        if not getattr(self, "_check_route", False):
            return nullcontext()
        return torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ])

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("size", [0, 1, 31, 32, 33, 97])
    def test_chunked_count_nonzero(self, device, dtype, size):
        values = torch.zeros(size + 2, device=device, dtype=dtype)
        values[0], values[-1] = 9, 9
        view = values[1:-1]
        if size:
            view[::3] = 1
            view[-1] = float("nan")
        self.assertEqual(_count_nonzero_chunked(view, 32), view.count_nonzero().item())
        self.assertEqual(values[[0, -1]], torch.full((2,), 9, device=device, dtype=dtype))
        with self.assertRaisesRegex(ValueError, "positive"):
            _count_nonzero_chunked(view, 0)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            _count_nonzero_chunked(torch.zeros((4, 8), device=device, dtype=dtype).t())

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
        with DeterministicGuard(deterministic), self._capture_pair_profile(device) as prof:
            output.index_add_(-2, index, source)
        expected = torch.zeros((3, 7, 5)).index_add(1, index.cpu(), source.float().cpu()).to(dtype)
        self.assertEqual(output.cpu().view(torch.int16), expected.view(torch.int16))
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertTrue(kernels, "CUDA profiling must capture device events")
            self.assertFalse(any("ppu_scatter_add_opt::" in name for name in kernels),
                             "index_add must not launch scatter_add optimization kernels")
            hit = any("ppu_index_add_opt::index_add_pair_" in name for name in kernels)
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
        with DeterministicGuard(False), self._capture_pair_profile(device) as prof:
            output.index_add_(-2, index, source)
        self.assertEqual(_count_nonzero_chunked(output), 7)
        self.assertEqual(output[0, index, 0], torch.full((7,), 0.125, device=device, dtype=dtype))
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertTrue(kernels, "CUDA profiling must capture device events")
            self.assertFalse(any("ppu_scatter_add_opt::" in name for name in kernels),
                             "index_add must not launch scatter_add optimization kernels")
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
        with DeterministicGuard(deterministic), self._capture_pair_profile(device) as prof:
            output.index_add_(dim, index, source, alpha=alpha)
        expected = initial.float().index_add(dim, index.cpu(), values.float(), alpha=alpha).to(dtype)
        self.assertEqual(output.cpu().view(torch.int32 if dtype == torch.float32 else torch.int16),
                         expected.view(torch.int32 if dtype == torch.float32 else torch.int16))
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertTrue(kernels, "CUDA profiling must capture device events")
            self.assertFalse(any("ppu_scatter_add_opt::" in name for name in kernels),
                             "index_add must not launch scatter_add optimization kernels")
            paired = [name for name in kernels if "ppu_index_add_opt::index_add_pair_" in name]
            expected_hit = (_INDEX_ADD_OPT_ENABLED and dtype != torch.float32 and not deterministic and dim == 0 and
                            layouts == ("contiguous", "contiguous") and offset == 0)
            self.assertEqual(bool(paired), expected_hit)
            if not paired and not deterministic:
                self.assertRegex(" ".join(kernels), r"indexFunc(?:Small|Large)Index")
            if paired and index_dtype == torch.int32:
                self.assertRegex(" ".join(paired), r"index_add_pair_kernel<[^,]+,\s*[^,]+,\s*(?:int|int32_t)\s*,")

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
        with DeterministicGuard(False), self._capture_pair_profile(device) as prof:
            output.index_add_(-2, index, source)
        expected = torch.zeros((2, 3, 1)).index_add(1, index.cpu(), torch.full((2, 5, 1), 0.125)).to(dtype)
        self.assertEqual(output.cpu().view(torch.int16), expected.view(torch.int16))
        self.assertEqual(source, torch.full((2, 5, 1), 0.125, device=device, dtype=dtype))
        self.assertEqual(storage[gaps].view(torch.int16), preserved.view(torch.int16))
        self.assertEqual(_count_nonzero_chunked(storage),
                         (6 if large_tensor == "output" else 10) + len(gap_offsets))
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertTrue(kernels, "CUDA profiling must capture device events")
            self.assertFalse(any("ppu_scatter_add_opt::" in name for name in kernels),
                             "index_add must not launch scatter_add optimization kernels")
            self.assertRegex(" ".join(kernels),
                             r"indexFunc(?:Small|Large)Index<[^,]+,\s*[^,]+,\s*(?:unsigned long(?: long)?|uint64_t)\s*,")

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("layout", ["contiguous", "strided", "broadcast"])
    def test_index_add_pair_rank_boundary_fallback(self, device, dtype, index_dtype, layout):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        shape = (1,) * 12 + (7,) + (1,) * 12
        source_shape = (1,) * 12 + (11,) + (1,) * 12
        values = torch.full(source_shape, 0.125, dtype=dtype)
        source, preserved, _ = self._make_index_add_pair_layout(values, device, layout, 12, 1)
        source_before = preserved.clone()
        output = torch.zeros(shape, device=device, dtype=dtype)
        output_before = output.clone()
        index = (torch.arange(11, device=device, dtype=index_dtype) * 3) % 7
        index_before = index.clone()
        with DeterministicGuard(False), self._capture_pair_profile(device) as prof:
            with self.assertRaisesRegex(RuntimeError, "CUDA Tensors cannot have more than 25 dimensions"):
                output.index_add_(-13, index, source)
            torch.cuda.synchronize()
        self.assertEqual(output.view(torch.int16), output_before.view(torch.int16))
        self.assertEqual(preserved.view(torch.int16), source_before.view(torch.int16))
        self.assertEqual(index, index_before)
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertFalse(any("ppu_scatter_add_opt::" in name for name in kernels),
                             "index_add must not launch scatter_add optimization kernels")
            self.assertFalse(any("ppu_index_add_opt::" in name for name in kernels),
                             "rank-boundary fallback must not launch index_add optimization kernels")

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("disable", [None, "0", "1"])
    @parametrize("other_disable", [None, "1"])
    def test_index_add_pair_disable_env(self, device, dtype, disable, other_disable):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        env = os.environ.copy()
        for name in ("PYTORCH_DISABLE_PPU_INDEX_ADD_OPT", "PYTORCH_DISABLE_PPU_SCATTER_ADD_OPT"):
            env.pop(name, None)
        if disable is not None:
            env["PYTORCH_DISABLE_PPU_INDEX_ADD_OPT"] = disable
        if other_disable is not None:
            env["PYTORCH_DISABLE_PPU_SCATTER_ADD_OPT"] = other_disable
        env["PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT"] = "0"
        env["PPU_ATOMIC_ROUTE_CHECK"] = str(int(self._check_route))
        script = _SUBPROCESS_TRACING + textwrap.dedent(r"""
            import os
            import sys
            import torch

            device = sys.argv[1]
            dtype = getattr(torch, sys.argv[2])
            enabled = sys.argv[3] == "1"
            torch.use_deterministic_algorithms(False)
            source = torch.full((11, 8), 0.125, device=device, dtype=dtype)
            index = torch.arange(11, device=device) * 3 % 7
            source_before, index_before = source.clone(), index.clone()
            expected = torch.zeros((7, 8)).index_add(
                0, index.cpu(), torch.full((11, 8), 0.125)).to(dtype)
            for iteration in range(2):
                output = torch.zeros((7, 8), device=device, dtype=dtype)
                with capture() as prof:
                    output.index_add_(0, index, source)
                    torch.cuda.synchronize()
                assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))
                assert torch.equal(source.view(torch.int16), source_before.view(torch.int16))
                assert torch.equal(index, index_before)
                kernels = device_kernels(prof)
                if kernels is not None:
                    assert kernels, "CUDA profiling must capture device events"
                    hit = any("ppu_index_add_opt::index_add_pair_kernel" in name for name in kernels)
                    assert hit == enabled, kernels
                    assert not any("ppu_scatter_add_opt::" in name for name in kernels), kernels
                    if not enabled:
                        assert any("indexFuncSmallIndex" in name or "indexFuncLargeIndex" in name
                                   for name in kernels), kernels
                if iteration == 0:
                    os.environ["PYTORCH_DISABLE_PPU_INDEX_ADD_OPT"] = "1" if enabled else "0"
            print("PASS")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script, str(device), str(dtype).split(".")[-1], str(int(disable != "1"))],
            env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)

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

    def _profile_index_add_pair(self, output, index, source, dim=0, alpha=1, deterministic=False):
        with DeterministicGuard(deterministic), self._capture_pair_profile(output.device) as prof:
            output.index_add_(dim, index, source, alpha=alpha)
            torch.cuda.synchronize(output.device)
        return _device_kernels(prof)

    def _assert_index_add_pair_route(self, kernels, hit, index_dtype=None, offset_type=None):
        if kernels is None:
            return []
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        self.assertFalse(any("ppu_scatter_add_opt::" in name for name in kernels))
        paired = [name for name in kernels if "ppu_index_add_opt::index_add_pair_kernel" in name]
        self.assertEqual(bool(paired), hit)
        if paired:
            index_type = r"(?:int|int32_t)" if index_dtype == torch.int32 else r"(?:long(?: long)?|int64_t)"
            offset = offset_type or r"[^,]+"
            self.assertRegex(" ".join(paired), rf"index_add_pair_kernel<[^,]+,\s*{offset},\s*{index_type}\s*,")
        elif not hit:
            self.assertRegex(" ".join(kernels), r"indexFunc(?:Small|Large)Index")
        return paired

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("alpha", [1, 0.1, -0.3, 1.0006, 3.141592653589793])
    def test_index_add_pair_fastpath_alpha_rounding(self, device, dtype, index_dtype, alpha):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        # One contribution per destination makes the oracle independent of atomic ordering.
        values = ((torch.arange(88).reshape(11, 8) % 127 - 63).float() / 13).to(dtype)
        values[0, :4] = torch.tensor([1, 1 + torch.finfo(dtype).eps / 2,
                                     1 + torch.finfo(dtype).eps, -1], dtype=dtype)
        values[0, 4:] = torch.tensor([torch.finfo(dtype).tiny / 2, torch.finfo(dtype).tiny,
                                     torch.finfo(dtype).max / 2, -torch.finfo(dtype).max / 2], dtype=dtype)
        initial = torch.full((11, 8), 0.125, dtype=dtype)
        ids = torch.arange(10, -1, -1, dtype=index_dtype)
        scaled = (values.float() * torch.tensor(alpha, dtype=dtype).float()).to(dtype)
        expected = initial.float().index_add(0, ids, scaled.float()).to(dtype)
        source, index, output = values.to(device), ids.to(device), initial.to(device)
        source_before, index_before = source.clone(), index.clone()
        kernels = self._profile_index_add_pair(output, index, source, alpha=alpha)
        paired = self._assert_index_add_pair_route(kernels, _INDEX_ADD_OPT_ENABLED, index_dtype)
        if paired:
            self.assertRegex(" ".join(paired), rf",\s*(?:{'false|0' if alpha == 1 else 'true|1'})\s*>")
        self.assertEqual(output.cpu().view(torch.int16), expected.view(torch.int16))
        self.assertEqual(source.view(torch.int16), source_before.view(torch.int16))
        self.assertEqual(index, index_before)

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("value", [1, 2])
    def test_index_add_pair_symbolic_alpha(self, device, dtype, value):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        shape_env = ShapeEnv(specialize_zero_one=False)
        symbol = shape_env.create_symbol(value, ConstantSource("ppu_index_add_alpha"), positive=None)
        alpha = shape_env.create_symintnode(symbol, hint=value)
        self.assertIsInstance(alpha, torch.SymInt)
        source = torch.full((2, 8), 0.125, device=device, dtype=dtype)
        output = torch.zeros_like(source)
        index = torch.arange(2, device=device)
        kernels = self._profile_index_add_pair(output, index, source, alpha=alpha)
        paired = self._assert_index_add_pair_route(kernels, _INDEX_ADD_OPT_ENABLED, torch.int64)
        if paired:
            self.assertRegex(" ".join(paired), r",\s*(?:true|1)\s*>")
        self.assertEqual(output, torch.full_like(output, 0.125 * value))

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("dim", [0, -2])
    @parametrize("deterministic", [False, True])
    def test_index_select_pair_fastpath_backward(self, device, dtype, index_dtype, dim, deterministic):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        leaf = torch.zeros((7, 8), device=device, dtype=dtype, requires_grad=True)
        grad = torch.full((11, 8), 0.125, device=device, dtype=dtype)
        index = torch.arange(11, device=device, dtype=index_dtype) * 3 % 7
        expected = torch.zeros((7, 8)).index_add(0, index.cpu(), grad.float().cpu()).to(dtype)
        with DeterministicGuard(deterministic), self._capture_pair_profile(device) as prof:
            leaf.index_select(dim, index).backward(grad)
            torch.cuda.synchronize(device)
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertFalse(any("ppu_scatter_add_opt::" in name for name in kernels))
            self.assertEqual(any("ppu_index_add_opt::index_add_pair_kernel" in name for name in kernels),
                             _INDEX_ADD_OPT_ENABLED and not deterministic)
            self.assertTrue(kernels)
        self.assertEqual(leaf.grad.cpu().view(torch.int16), expected.view(torch.int16))

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("case", ["base", "output_offset1", "source_offset1", "offset2",
                          "output_transpose", "source_transpose", "output_strided",
                          "source_strided", "source_broadcast", "index_strided", "index_broadcast"])
    def test_index_add_pair_single_guard_dispatch(self, device, dtype, index_dtype, case):
        values = torch.full((11, 8), 0.125, dtype=dtype)
        initial = torch.full((7, 8), 0.125, dtype=dtype)
        source_layout = case.removeprefix("source_") if case.startswith("source_") else "contiguous"
        output_layout = case.removeprefix("output_") if case.startswith("output_") else "contiguous"
        source_offset = 1 if case == "source_offset1" else 2 if case == "offset2" else 0
        output_offset = 1 if case == "output_offset1" else 2 if case == "offset2" else 0
        if "offset" in source_layout:
            source_layout = "contiguous"
        if "offset" in output_layout:
            output_layout = "contiguous"
        source, _, _ = self._make_index_add_pair_layout(values, device, source_layout, 0, source_offset)
        output, _, _ = self._make_index_add_pair_layout(initial, device, output_layout, 0, output_offset)
        index = torch.arange(11, device=device, dtype=index_dtype) * 3 % 7
        if case == "index_strided":
            index = index.repeat_interleave(2)[::2]
        elif case == "index_broadcast":
            index = torch.zeros(1, device=device, dtype=index_dtype).expand(11)
        expected = initial.float().index_add(0, index.cpu(), values.float()).to(dtype)
        kernels = self._profile_index_add_pair(output, index, source)
        eligible = bool(getattr(torch.version, "ppu", None)) and torch.cuda.get_device_capability(device) == (8, 9)
        hit = eligible and _INDEX_ADD_OPT_ENABLED and case in ("base", "offset2", "index_strided", "index_broadcast")
        self._assert_index_add_pair_route(kernels, hit, index_dtype)
        self.assertEqual(output.cpu().view(torch.int16), expected.view(torch.int16))

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("large_tensor", ["output", "source"])
    @serialTest()
    @largeTensorTest("10GB")
    def test_index_add_pair_fastpath_64bit(self, device, dtype, index_dtype, large_tensor):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        if large_tensor == "output":
            rows = (1 << 28) + 1
            output = torch.zeros((rows, 8), device=device, dtype=dtype)
            source = torch.full((2, 8), 0.125, device=device, dtype=dtype)
            index = torch.tensor([0, rows - 1], device=device, dtype=index_dtype)
            self.assertGreater(output.numel(), torch.iinfo(torch.int32).max)
        else:
            width = (1 << 30) + 2
            output = torch.zeros((1, width), device=device, dtype=dtype)
            source = torch.full((2, width), 0.125, device=device, dtype=dtype)
            index = torch.zeros(2, device=device, dtype=index_dtype)
            self.assertGreater(source.numel(), torch.iinfo(torch.int32).max)
            self.assertLess(output.numel(), torch.iinfo(torch.int32).max)
        kernels = self._profile_index_add_pair(output, index, source)
        self._assert_index_add_pair_route(kernels, _INDEX_ADD_OPT_ENABLED, index_dtype,
                                         r"(?:unsigned long(?: long)?|uint64_t)")
        if large_tensor == "output":
            self.assertEqual(output[[0, rows - 1]], torch.full_like(source, 0.125))
            self.assertEqual(_count_nonzero_chunked(output), 16)
        else:
            self.assertEqual(output.amin().item(), 0.25)
            self.assertEqual(output.amax().item(), 0.25)
        self.assertEqual(source.amin().item(), 0.125)
        self.assertEqual(source.amax().item(), 0.125)

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @serialTest()
    @largeTensorTest("20GB")
    def test_index_add_pair_index_stride_64bit(self, device, dtype, index_dtype):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        span = (1 << 31) + 1
        storage = torch.empty(span + 1, device=device, dtype=index_dtype)
        index = storage.as_strided((2,), (span,))
        index.copy_(torch.tensor([0, 1], device=device, dtype=index_dtype))
        output = torch.zeros((2, 8), device=device, dtype=dtype)
        source = torch.full_like(output, 0.125)
        kernels = self._profile_index_add_pair(output, index, source)
        self._assert_index_add_pair_route(kernels, _INDEX_ADD_OPT_ENABLED, index_dtype,
                                         r"(?:unsigned long(?: long)?|uint64_t)")
        self.assertEqual(output, source)
        self.assertEqual(index.cpu(), torch.tensor([0, 1], dtype=index_dtype))

    def _make_index_add_special_case(self, dtype, layout):
        info = torch.finfo(dtype)
        row = [-0.0, info.tiny / 2, info.max, float("inf"),
               -float("inf"), float("nan"), 1.0, -1.0]
        source = torch.tensor([row, row], dtype=dtype)
        initial = torch.zeros((2, 8), dtype=dtype)
        initial[:, 6], initial[:, 7] = -1.0, 1.0
        index = torch.tensor([1, 0])
        if layout == "strided":
            index = index.repeat_interleave(2)[::2]
        elif layout == "broadcast":
            # Two identical contributions remain order independent, including NaN/Inf.
            index = torch.zeros(1, dtype=torch.int64).expand(2)
        expected = initial.float().index_add(0, index, source.float()).to(dtype)
        return source, index, initial, expected

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_layout", ["dense", "strided", "broadcast"])
    def test_index_add_pair_special_values(self, device, dtype, index_layout):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        source, ids, initial, expected = self._make_index_add_special_case(dtype, index_layout)
        output, source, index = initial.to(device), source.to(device), ids.to(device)
        if index_layout == "strided":
            index = index.repeat_interleave(2)[::2]
        elif index_layout == "broadcast":
            index = torch.zeros(1, device=device, dtype=torch.int64).expand(2)
        source_before, index_before = source.clone(), index.clone()
        kernels = self._profile_index_add_pair(output, index, source)
        self._assert_index_add_pair_route(kernels, _INDEX_ADD_OPT_ENABLED, torch.int64)
        self.assertEqual(source.view(torch.int16), source_before.view(torch.int16))
        self.assertEqual(index, index_before)
        actual = output.cpu()
        self.assertEqual(actual.isnan(), expected.isnan())
        valid = ~expected.isnan()
        self.assertEqual(actual.view(torch.int16)[valid], expected.view(torch.int16)[valid])

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", ["empty_index", "empty_inner", "bad_shape", "bad_dtype", "overlap", "rank24", "rank25", "rank26"])
    def test_index_add_pair_boundary_contract(self, device, dtype, case):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        output = torch.zeros((2, 8), device=device, dtype=dtype)
        source = torch.full_like(output, 0.125)
        index = torch.arange(2, device=device)
        error = None
        if case == "empty_index":
            index, source = index[:0], source[:0]
        elif case == "empty_inner":
            output, source = output[:, :0], source[:, :0]
        elif case == "bad_shape":
            source = source[:1]
            error = "Number of indices|source.size|same size"
        elif case == "bad_dtype":
            source = source.float()
            error = "same scalar type|same dtype"
        elif case == "overlap":
            source = output
            error = "single memory location|overlap"
        elif case.startswith("rank"):
            rank = int(case[4:])
            output = output.reshape((2,) + (1,) * (rank - 2) + (8,))
            source = source.reshape(output.shape)
            if rank == 26:
                error = "tensor has too many"
            elif rank == 25 and not _INDEX_ADD_OPT_ENABLED:
                error = "CUDA Tensors cannot have more than 25 dimensions"
        before = output.clone()
        with DeterministicGuard(False), self._capture_pair_profile(device) as prof:
            if error:
                with self.assertRaisesRegex(RuntimeError, error):
                    output.index_add_(0, index, source)
            else:
                output.index_add_(0, index, source)
            torch.cuda.synchronize(device)
        kernels = _device_kernels(prof)
        hit = case in ("rank24", "rank25") and not error and _INDEX_ADD_OPT_ENABLED
        if kernels is not None:
            if hit:
                self.assertTrue(kernels, "CUDA profiling must capture device events")
            self.assertEqual(any("ppu_index_add_opt::" in name for name in kernels), hit)
        if error or case.startswith("empty"):
            self.assertEqual(output.view(torch.int16), before.view(torch.int16))
        else:
            self.assertEqual(output, torch.full_like(output, 0.125))

    @onlyOn(["cuda"])
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("first", ["index_add", "scatter_add"])
    @parametrize("scenario", ["late_import", "miss_cache", "alternating", "invalid"])
    def test_paired_atomic_env_initialization(self, device, dtype, first, scenario):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        env = os.environ.copy()
        for op in ("INDEX_ADD", "SCATTER_ADD"):
            env.pop(f"PYTORCH_DISABLE_PPU_{op}_OPT", None)
        env["PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT"] = "0"
        env["PPU_ATOMIC_ROUTE_CHECK"] = str(int(self._check_route))
        script = _SUBPROCESS_TRACING + textwrap.dedent(r"""
            import os
            import sys
            import warnings
            import torch
            device, dtype = sys.argv[1], getattr(torch, sys.argv[2])
            first, scenario = sys.argv[3:5]
            second = "scatter_add" if first == "index_add" else "index_add"
            torch.cuda.set_device(device)
            torch.use_deterministic_algorithms(False)
            def flag(op):
                return f"PYTORCH_DISABLE_PPU_{op.upper()}_OPT"
            def run(op, enabled, miss=False):
                output = torch.zeros((2, 8), device=device, dtype=dtype)
                source = torch.full_like(output, 0.125)
                index = torch.arange(2, device=device)
                if op == "scatter_add":
                    index = index.reshape(2, 1).expand_as(source)
                if miss:
                    output, source = output.t().contiguous(), source.t().contiguous()
                    if op == "scatter_add":
                        index = index.t()
                with capture() as prof:
                    getattr(output, op + "_")(1 if miss else 0, index, source)
                    torch.cuda.synchronize(device)
                kernels = device_kernels(prof)
                if kernels is not None:
                    assert kernels, "CUDA profiling must capture device events"
                    hit = any(f"ppu_{op}_opt::{op}_pair_kernel" in k for k in kernels)
                    assert hit == (enabled and not miss), kernels
                    assert not any(f"ppu_{second if op == first else first}_opt::" in k for k in kernels), kernels
                    if not hit:
                        native = "indexFunc" if op == "index_add" else "scatter_gather"
                        assert any(native in k for k in kernels), kernels
                assert torch.equal(output, torch.full_like(output, 0.125))
            # torch has already been imported; neither optimization entry has run yet.
            if scenario == "invalid":
                os.environ[flag(first)], os.environ[flag(second)] = "true", "2"
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    run(first, True)
                    run(second, True)
                    messages = [str(w.message) for w in caught]
                    for op in (first, second):
                        assert sum("Ignoring invalid value" in m and flag(op) in m for m in messages) == 1, messages
                    os.environ[flag(first)] = os.environ[flag(second)] = "1"
                    run(first, True)
                    run(second, True)
                    assert len(caught) == len(messages)
            else:
                os.environ[flag(first)], os.environ[flag(second)] = "1", "0"
                if scenario == "miss_cache":
                    run(first, False, miss=True)
                    run(second, True, miss=True)
                else:
                    run(first, False)
                    if scenario == "late_import":
                        os.environ[flag(second)] = "1"
                        run(second, False)
                        os.environ[flag(second)] = "0"
                        run(second, False)
                        print("PASS")
                        sys.exit(0)
                    run(second, True)
                os.environ[flag(first)], os.environ[flag(second)] = "0", "1"
                run(first, False)
                run(second, True)
            print("PASS")
        """)
        result = subprocess.run([sys.executable, "-c", script, str(device), str(dtype).split(".")[-1], first, scenario],
                                env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)

    @onlyOn(["cuda"])
    @dtypes(torch.float16)
    @parametrize("bad", [-1, 2])
    def test_index_add_pair_invalid_index_subprocess(self, device, dtype, bad):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(device) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        script = textwrap.dedent(r"""
            import sys
            import torch
            device = sys.argv[1]
            torch.cuda.set_device(device)
            torch.use_deterministic_algorithms(False)
            try:
                torch.zeros((2, 8), device=device, dtype=torch.float16).index_add_(
                    0, torch.tensor([0, int(sys.argv[2])], device=device),
                    torch.ones((2, 8), device=device, dtype=torch.float16))
                torch.cuda.synchronize(device)
            except RuntimeError as error:
                assert "device-side assert" in str(error) or "out of bounds" in str(error), str(error)
                print("EXPECTED_DEVICE_ASSERT")
                sys.exit(0)
            raise AssertionError("An invalid index must not complete successfully")
        """)
        result = subprocess.run([sys.executable, "-c", script, str(device), str(bad)],
                                env=os.environ.copy(), capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("EXPECTED_DEVICE_ASSERT", result.stdout)


# Set PPU_ATOMIC_FULL_MATRIX=1 to restore the exhaustive Cartesian products.
for _name in ("test_index_add_pairs_nd", "test_index_add_pairs_layouts",
              "test_index_add_pairs_layout_dispatch"):
    setattr(TestPPUIndexAdd, _name, _representative_matrix(getattr(TestPPUIndexAdd, _name)))

# Numerical and tracing cases are collected independently, including subprocess tests.
for _name in (
    "test_index_add_pair_nd_dispatch", "test_index_add_pairs_nd_64bit_addressing",
    "test_index_add_pairs_layout_dispatch", "test_index_add_pairs_layout_64bit",
    "test_index_add_pair_rank_boundary_fallback", "test_index_add_pair_disable_env",
    "test_index_add_pair_fastpath_alpha_rounding", "test_index_add_pair_symbolic_alpha",
    "test_index_select_pair_fastpath_backward", "test_index_add_pair_single_guard_dispatch",
    "test_index_add_pair_fastpath_64bit", "test_index_add_pair_index_stride_64bit",
    "test_index_add_pair_special_values", "test_index_add_pair_boundary_contract",
    "test_paired_atomic_env_initialization",
):
    setattr(TestPPUIndexAdd, _name, _with_route_check(getattr(TestPPUIndexAdd, _name)))

instantiate_device_type_tests(TestPPUIndexAdd, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
