# Owner(s): ["module: scatter & gather ops"]
# PPU paired-atomic scatter_add coverage, including native fallback sentinels.
# Run: python -m pytest test/ppu_tests/test_scatter_add.py

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
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    parametrize,
    run_tests,
    serialTest,
    TestCase,
)


_SCATTER_ADD_OPT_ENABLED = os.getenv("PYTORCH_DISABLE_PPU_SCATTER_ADD_OPT") != "1"


def _count_nonzero_chunked(tensor, chunk_size=1 << 20):
    # Keep bool/int64 reduction temporaries bounded without copying a large view.
    if not tensor.is_contiguous():
        raise ValueError("Chunked counting requires a contiguous tensor")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    flat = tensor.view(-1)
    return sum(
        flat[start : start + chunk_size].count_nonzero().item()
        for start in range(0, flat.numel(), chunk_size)
    )


def _representative_matrix(fn):
    if os.getenv("PPU_ATOMIC_FULL_MATRIX") == "1":
        return fn
    full_parametrize = fn.parametrize_fn

    def representative(test, generic_cls, device_cls):
        cases = list(full_parametrize(test, generic_cls, device_cls))
        anchor = cases[0][2]
        groups = [
            (name,) for name in anchor if name not in ("index_dtype", "shape", "dim")
        ]
        groups.append(tuple(name for name in ("shape", "dim") if name in anchor))
        # Treat coupled shape/dim parameters as one factor; retain both index dtypes.
        for case in cases:
            differences = sum(
                any(case[2][name] != anchor[name] for name in group) for group in groups
            )
            if differences <= 2:
                yield case

    fn.parametrize_fn = representative
    return fn


_TRACING_DEVICES = set()


def _device_kernels(prof):
    if prof is None:
        return None
    return [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]


def _require_kernel_tracing(device):
    if (
        torch.profiler.ProfilerActivity.CUDA
        not in torch.profiler.supported_activities()
    ):
        raise unittest.SkipTest(
            "CUDA kernel tracing is unavailable; numerical cases run separately"
        )
    key = str(device)
    if key not in _TRACING_DEVICES:
        # Probe an unrelated kernel before either operator's first-entry cache is initialized.
        probe = torch.ones(8, device=device)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            probe.add_(1)
            torch.cuda.synchronize(device)
        if not _device_kernels(prof):
            raise AssertionError(
                "CUDA tracing was advertised but the device-kernel probe captured no events"
            )
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


class TestPPUScatterAdd(TestCase):
    def _capture_pair_profile(self, device):
        if not getattr(self, "_check_route", False):
            return nullcontext()
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )

    @onlyCUDA
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
        self.assertEqual(
            values[[0, -1]], torch.full((2,), 9, device=device, dtype=dtype)
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            _count_nonzero_chunked(view, 0)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            _count_nonzero_chunked(torch.zeros((4, 8), device=device, dtype=dtype).t())

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize(
        "shape", [(), (7,), (7, 6), (3, 7, 5), (2, 3, 7, 4), (2, 1, 3, 7, 3), (1,) * 24]
    )
    @parametrize("index_layout", ["broadcast", "dense", "strided", "permuted"])
    @parametrize("partial", [False, True])
    @parametrize("offsets", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_scatter_add_pairs_nd(
        self, device, dtype, index_dtype, shape, index_layout, partial, offsets
    ):
        rank = len(shape)
        dims = (0, rank // 2, rank - 1) if rank > 5 else range(max(1, rank))
        for d in dims:
            output_shape, source_shape = list(shape), list(shape)
            if rank:
                output_shape[d], source_shape[d] = 7, 11
                if partial:
                    source_shape = [
                        s + 2 if i != d and s > 1 else s
                        for i, s in enumerate(source_shape)
                    ]
            index_shape = list(source_shape)
            if partial and rank:
                index_shape = [
                    9 if i == d else max(1, output_shape[i] - 1) for i in range(rank)
                ]
            source_numel = reduce(operator.mul, source_shape, 1)
            output_numel = reduce(operator.mul, output_shape, 1)
            index_numel = reduce(operator.mul, index_shape, 1)
            source_cpu = (
                (torch.arange(source_numel).reshape(source_shape) % 7 - 3).float()
                * 0.125
            ).to(dtype)
            initial = torch.full(output_shape, 0.125, dtype=dtype)
            if rank and index_layout == "broadcast":
                index_view = [1] * rank
                index_view[d] = index_shape[d]
                ids = (torch.arange(index_shape[d], dtype=index_dtype) * 3) % 7
                index_cpu = ids.reshape(index_view).expand(index_shape)
            else:
                index_cpu = (
                    (torch.arange(index_numel, dtype=index_dtype) * 3) % 7
                ).reshape(index_shape)
            expected = (
                initial.float()
                .scatter_add(d, index_cpu.long(), source_cpu.float())
                .to(dtype)
            )
            for dim in (d, d - max(1, rank)):
                output_offset, source_offset = offsets
                source_storage = torch.full(
                    (source_numel + 4,), 17.0, device=device, dtype=dtype
                )
                source = source_storage[
                    source_offset : source_offset + source_numel
                ].reshape(source_shape)
                source.copy_(source_cpu)
                preserved_source = source_storage.clone()
                output_storage = torch.full(
                    (output_numel + 4,), 9.0, device=device, dtype=dtype
                )
                output = output_storage[
                    output_offset : output_offset + output_numel
                ].reshape(output_shape)
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
                    functional = torch.scatter_add(
                        initial.to(device), dim, index, source
                    )
                    out = torch.empty_like(output)
                    torch.scatter_add(initial.to(device), dim, index, source, out=out)
                bits = torch.int32 if dtype == torch.float32 else torch.int16
                for actual in (output, functional, out):
                    self.assertEqual(actual.cpu().view(bits), expected.view(bits))
                self.assertEqual(source_storage, preserved_source)
                self.assertEqual(index.cpu(), index_before)
                self.assertEqual(
                    output_storage[:output_offset],
                    torch.full_like(output_storage[:output_offset], 9.0),
                )
                tail = output_storage[output_offset + output_numel :]
                self.assertEqual(tail, torch.full_like(tail, 9.0))

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize(
        "shape, dim",
        [
            ((7,), 0),
            ((7, 8), 0),
            ((7, 3, 8), 0),
            ((3, 7, 5), 1),
            ((2, 3, 7, 4), 2),
            ((3, 7, 5), -1),
            ((1,) * 12 + (7,) + (1,) * 11, 12),
        ],
    )
    @parametrize("deterministic", [False, True])
    def test_gather_pair_backward_nd(
        self, device, dtype, index_dtype, shape, dim, deterministic
    ):
        d = dim % len(shape)
        grad_shape, index_view = list(shape), [1] * len(shape)
        grad_shape[d], index_view[d] = 11, 11
        ids = (torch.arange(11, dtype=index_dtype) * 3) % shape[d]
        index_cpu = ids.reshape(index_view).expand(grad_shape)
        numel = reduce(operator.mul, grad_shape, 1)
        grad_cpu = (
            (torch.arange(numel).reshape(grad_shape) % 7 - 3).float() * 0.125
        ).to(dtype)
        expected = (
            torch.zeros(shape)
            .scatter_add(dim, index_cpu.long(), grad_cpu.float())
            .to(dtype)
        )
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
    def test_scatter_add_pair_nd_dispatch(
        self, device, dtype, index_dtype, index_layout, deterministic, offset
    ):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        shape = (3, 7, 5)
        source_shape = (5, 11, 7) if index_layout == "partial" else (3, 11, 5)
        index_shape = (2, 9, 4) if index_layout == "partial" else source_shape
        source = torch.full(source_shape, 0.125, device=device, dtype=dtype)
        storage = torch.zeros(3 * 7 * 5 + 2, device=device, dtype=dtype)
        output = storage[offset : offset + 3 * 7 * 5].reshape(shape)
        if index_layout == "broadcast":
            index = (
                (torch.arange(11, device=device, dtype=index_dtype) % 7)
                .reshape(1, 11, 1)
                .expand(index_shape)
            )
        else:
            numel = reduce(operator.mul, index_shape, 1)
            index = (torch.arange(numel, device=device, dtype=index_dtype) % 7).reshape(
                index_shape
            )
            if index_layout == "strided":
                index = index.repeat_interleave(2, dim=-1)[..., ::2]
        with (
            DeterministicGuard(deterministic),
            self._capture_pair_profile(device) as prof,
        ):
            output.scatter_add_(-2, index, source)
        expected = (
            torch.zeros(shape)
            .scatter_add(1, index.long().cpu(), source.float().cpu())
            .to(dtype)
        )
        self.assertEqual(output.cpu().view(torch.int16), expected.view(torch.int16))
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertTrue(kernels, "CUDA profiling must capture device events")
            self.assertFalse(
                any("ppu_index_add_opt::" in name for name in kernels),
                "scatter_add must not launch index_add optimization kernels",
            )
            hit = any(
                "ppu_scatter_add_opt::scatter_add_pair_" in name for name in kernels
            )
            self.assertFalse(hit)
            if not deterministic:
                self.assertRegex(
                    " ".join(kernels), r"scatter_gather|vectorized_scatter_add_kernel"
                )

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @serialTest()
    @largeTensorTest("5GB")
    def test_scatter_add_pairs_nd_64bit_addressing(self, device, dtype):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        size = (1 << 31) + 5
        output = torch.zeros((1, size, 1), device=device, dtype=dtype)
        source = torch.full((2, 7, 1), 0.125, device=device, dtype=dtype)
        index = torch.tensor(
            [size - 1, size - 2, 0, 1, 2, 3, 4], device=device
        ).reshape(1, 7, 1)
        with DeterministicGuard(False), self._capture_pair_profile(device) as prof:
            output.scatter_add_(-2, index, source)
        self.assertEqual(_count_nonzero_chunked(output), 7)
        self.assertEqual(
            output[0, index.reshape(-1), 0],
            torch.full((7,), 0.125, device=device, dtype=dtype),
        )
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertTrue(kernels, "CUDA profiling must capture device events")
            self.assertFalse(
                any("ppu_index_add_opt::" in name for name in kernels),
                "scatter_add must not launch index_add optimization kernels",
            )
            self.assertFalse(
                any(
                    "ppu_scatter_add_opt::scatter_add_pair_" in name for name in kernels
                )
            )
            self.assertRegex(
                " ".join(kernels), r"scatter_gather|vectorized_scatter_add_kernel"
            )

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize(
        "shape, dim", [((11, 8), 0), ((11, 3, 8), 0), ((3, 11, 8), 1), ((11, 7), 0)]
    )
    @parametrize(
        "index_layout", ["broadcast", "dense_same", "dense_mixed", "strided", "partial"]
    )
    @parametrize("offset", [0, 1])
    @parametrize("deterministic", [False, True])
    def test_scatter_add_pair_fastpath_dispatch(
        self,
        device,
        dtype,
        index_dtype,
        shape,
        dim,
        index_layout,
        offset,
        deterministic,
    ):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        output_shape = list(shape)
        output_shape[dim] = 7
        source = torch.full(shape, 0.125, device=device, dtype=dtype)
        storage = torch.full((source.numel() + 4,), 9.0, device=device, dtype=dtype)
        source = storage[offset : offset + source.numel()].reshape(shape).copy_(source)
        output_storage = torch.full(
            (reduce(operator.mul, output_shape, 1) + 4,),
            9.0,
            device=device,
            dtype=dtype,
        )
        output = (
            output_storage[offset : offset + reduce(operator.mul, output_shape, 1)]
            .reshape(output_shape)
            .zero_()
        )
        ids_shape = [1] * len(shape)
        ids_shape[dim] = shape[dim]
        index = (
            (torch.arange(shape[dim], device=device, dtype=index_dtype) * 3 % 7)
            .reshape(ids_shape)
            .expand(shape)
        )
        if index_layout != "broadcast":
            index = index.contiguous()
        if index_layout == "dense_mixed":
            index = (
                index
                + torch.arange(index.numel(), device=device, dtype=index_dtype).reshape(
                    shape
                )
            ) % 7
        elif index_layout == "strided":
            index = index.repeat_interleave(2, dim=-1)[..., ::2]
        elif index_layout == "partial":
            index = index.narrow(dim, 0, 9)
        expected = (
            torch.zeros(output_shape)
            .scatter_add(dim, index.long().cpu(), source.float().cpu())
            .to(dtype)
        )
        source_before = storage.clone()
        with (
            DeterministicGuard(deterministic),
            self._capture_pair_profile(device) as prof,
        ):
            output.scatter_add_(dim, index, source)
            torch.cuda.synchronize()
        self.assertEqual(
            output.cpu().view(torch.int32 if dtype == torch.float32 else torch.int16),
            expected.view(torch.int32 if dtype == torch.float32 else torch.int16),
        )
        self.assertEqual(storage, source_before)
        self.assertEqual(
            output_storage[:offset], torch.full_like(output_storage[:offset], 9.0)
        )
        tail = output_storage[offset + output.numel() :]
        self.assertEqual(tail, torch.full_like(tail, 9.0))
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertTrue(kernels, "CUDA profiling must capture device events")
            self.assertFalse(
                any("ppu_index_add_opt::" in name for name in kernels),
                "scatter_add must not launch index_add optimization kernels",
            )
            paired = [
                name
                for name in kernels
                if "ppu_scatter_add_opt::scatter_add_pair_kernel" in name
            ]
            inner = reduce(operator.mul, shape[dim + 1 :], 1)
            expected_hit = (
                _SCATTER_ADD_OPT_ENABLED
                and dtype != torch.float32
                and not deterministic
                and dim == 0
                and inner % 2 == 0
                and offset == 0
                and index_layout in ("broadcast", "dense_same", "dense_mixed")
            )
            self.assertEqual(bool(paired), expected_hit)
            if paired:
                index_type = (
                    r"(?:int|int32_t)"
                    if index_dtype == torch.int32
                    else r"(?:long(?: long)?|int64_t)"
                )
                self.assertRegex(
                    " ".join(paired),
                    rf"scatter_add_pair_kernel<[^,]+,\s*[^,]+,\s*{index_type}\s*,",
                )
            if not expected_hit and not deterministic:
                self.assertRegex(
                    " ".join(kernels), r"scatter_gather|vectorized_scatter_add_kernel"
                )

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("index_layout", ["broadcast", "dense_same", "dense_mixed", "strided"])
    @parametrize("dim", [0, -2])
    def test_gather_pair_fastpath_backward_dispatch(
        self, device, dtype, index_dtype, index_layout, dim
    ):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        leaf = torch.zeros((7, 8), device=device, dtype=dtype, requires_grad=True)
        grad = torch.full((11, 8), 0.125, device=device, dtype=dtype)
        index = (
            (torch.arange(11, device=device, dtype=index_dtype) * 3 % 7)
            .reshape(11, 1)
            .expand_as(grad)
        )
        if index_layout != "broadcast":
            index = index.contiguous()
        if index_layout == "dense_mixed":
            index = (
                index
                + torch.arange(index.numel(), device=device, dtype=index_dtype).reshape(
                    grad.shape
                )
            ) % 7
        elif index_layout == "strided":
            index = index.repeat_interleave(2, dim=-1)[..., ::2]
        expected = (
            torch.zeros((7, 8))
            .scatter_add(dim, index.long().cpu(), grad.float().cpu())
            .to(dtype)
        )
        with DeterministicGuard(False), self._capture_pair_profile(device) as prof:
            leaf.gather(dim, index).backward(grad)
            torch.cuda.synchronize()
        self.assertEqual(leaf.grad.cpu().view(torch.int16), expected.view(torch.int16))
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertTrue(kernels, "CUDA profiling must capture device events")
            self.assertFalse(
                any("ppu_index_add_opt::" in name for name in kernels),
                "scatter_add must not launch index_add optimization kernels",
            )
            paired = [
                name
                for name in kernels
                if "ppu_scatter_add_opt::scatter_add_pair_kernel" in name
            ]
            self.assertEqual(
                bool(paired), _SCATTER_ADD_OPT_ENABLED and index_layout != "strided"
            )
            if paired:
                index_type = (
                    r"(?:int|int32_t)"
                    if index_dtype == torch.int32
                    else r"(?:long(?: long)?|int64_t)"
                )
                self.assertRegex(
                    " ".join(paired),
                    rf"scatter_add_pair_kernel<[^,]+,\s*[^,]+,\s*{index_type}\s*,",
                )
            else:
                self.assertRegex(
                    " ".join(kernels), r"scatter_gather|vectorized_scatter_add_kernel"
                )

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("index_layout", ["broadcast", "dense_same", "dense_mixed"])
    @serialTest()
    @largeTensorTest("5GB")
    def test_scatter_add_pair_fastpath_64bit_addressing(
        self, device, dtype, index_dtype, index_layout
    ):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        rows = (1 << 28) + 1
        output = torch.zeros((rows, 8), device=device, dtype=dtype)
        source = torch.full((2, 8), 0.125, device=device, dtype=dtype)
        index = (
            torch.tensor([0, rows - 1], device=device, dtype=index_dtype)
            .reshape(2, 1)
            .expand_as(source)
        )
        if index_layout != "broadcast":
            index = index.contiguous()
        if index_layout == "dense_mixed":
            index = (
                torch.arange(16, device=device, dtype=index_dtype) % 2 * (rows - 1)
            ).reshape(2, 8)
        local_index = (index.cpu() == rows - 1).to(index_dtype)
        expected = (
            torch.zeros((2, 8))
            .scatter_add(0, local_index.long(), source.float().cpu())
            .to(dtype)
        )
        source_before = source.clone()
        self.assertGreater(output.numel(), torch.iinfo(torch.int32).max)
        with DeterministicGuard(False), self._capture_pair_profile(device) as prof:
            output.scatter_add_(0, index, source)
            torch.cuda.synchronize()
        self.assertEqual(
            output[[0, rows - 1]].cpu().view(torch.int16), expected.view(torch.int16)
        )
        self.assertEqual(
            _count_nonzero_chunked(output), expected.count_nonzero().item()
        )
        self.assertEqual(source.view(torch.int16), source_before.view(torch.int16))
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertTrue(kernels, "CUDA profiling must capture device events")
            self.assertFalse(
                any("ppu_index_add_opt::" in name for name in kernels),
                "scatter_add must not launch index_add optimization kernels",
            )
            paired = [
                name
                for name in kernels
                if "ppu_scatter_add_opt::scatter_add_pair_kernel" in name
            ]
            self.assertEqual(bool(paired), _SCATTER_ADD_OPT_ENABLED)
            if paired:
                index_type = (
                    r"(?:int|int32_t)"
                    if index_dtype == torch.int32
                    else r"(?:long(?: long)?|int64_t)"
                )
                self.assertRegex(
                    " ".join(paired),
                    rf"scatter_add_pair_kernel<[^,]+,\s*(?:unsigned long(?: long)?|uint64_t),\s*{index_type}\s*,",
                )
            else:
                self.assertRegex(
                    " ".join(kernels), r"scatter_gather|vectorized_scatter_add_kernel"
                )

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("disable", [None, "0", "1"])
    @parametrize("other_disable", [None, "1"])
    @parametrize("index_layout", ["broadcast", "dense"])
    def test_scatter_add_pair_disable_env(
        self, device, dtype, disable, other_disable, index_layout
    ):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        env = os.environ.copy()
        for name in (
            "PYTORCH_DISABLE_PPU_SCATTER_ADD_OPT",
            "PYTORCH_DISABLE_PPU_INDEX_ADD_OPT",
        ):
            env.pop(name, None)
        if disable is not None:
            env["PYTORCH_DISABLE_PPU_SCATTER_ADD_OPT"] = disable
        if other_disable is not None:
            env["PYTORCH_DISABLE_PPU_INDEX_ADD_OPT"] = other_disable
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
            index = (torch.arange(11, device=device) * 3 % 7).reshape(11, 1).expand_as(source)
            if sys.argv[4] == "dense":
                index = (torch.arange(88, device=device) % 7).reshape(11, 8)
            source_before, index_before = source.clone(), index.clone()
            expected = torch.zeros((7, 8)).scatter_add(
                0, index.cpu(), torch.full((11, 8), 0.125)).to(dtype)
            for iteration in range(2):
                output = torch.zeros((7, 8), device=device, dtype=dtype)
                with capture() as prof:
                    output.scatter_add_(0, index, source)
                    torch.cuda.synchronize()
                assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))
                assert torch.equal(source.view(torch.int16), source_before.view(torch.int16))
                assert torch.equal(index, index_before)
                kernels = device_kernels(prof)
                if kernels is not None:
                    assert kernels, "CUDA profiling must capture device events"
                    hit = any("ppu_scatter_add_opt::scatter_add_pair_kernel" in name for name in kernels)
                    assert hit == enabled, kernels
                    assert not any("ppu_index_add_opt::" in name for name in kernels), kernels
                    if not enabled:
                        assert any("scatter_gather" in name or "vectorized_scatter_add_kernel" in name
                                   for name in kernels), kernels
                if iteration == 0:
                    os.environ["PYTORCH_DISABLE_PPU_SCATTER_ADD_OPT"] = "1" if enabled else "0"
            print("PASS")
        """)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(device),
                str(dtype).split(".")[-1],
                str(int(disable != "1")),
                index_layout,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)

    def _profile_scatter_add_pair(self, output, index, source, dim=0):
        with (
            DeterministicGuard(False),
            self._capture_pair_profile(output.device) as prof,
        ):
            output.scatter_add_(dim, index, source)
            torch.cuda.synchronize(output.device)
        return _device_kernels(prof)

    def _assert_scatter_add_pair_route(
        self, kernels, hit, index_dtype, offset_type=None
    ):
        if kernels is None:
            return []
        self.assertTrue(kernels, "CUDA profiling must capture device events")
        self.assertFalse(any("ppu_index_add_opt::" in name for name in kernels))
        paired = [
            name
            for name in kernels
            if "ppu_scatter_add_opt::scatter_add_pair_kernel" in name
        ]
        self.assertEqual(bool(paired), hit)
        if paired:
            index_type = (
                r"(?:int|int32_t)"
                if index_dtype == torch.int32
                else r"(?:long(?: long)?|int64_t)"
            )
            offset = offset_type or r"[^,]+"
            self.assertRegex(
                " ".join(paired),
                rf"scatter_add_pair_kernel<[^,]+,\s*{offset},\s*{index_type}\s*,",
            )
        else:
            self.assertRegex(
                " ".join(kernels), r"scatter_gather|vectorized_scatter_add_kernel"
            )

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize(
        "case",
        [
            "base",
            "output_offset1",
            "source_offset1",
            "offset2",
            "output_transpose",
            "source_transpose",
            "output_strided",
            "source_strided",
            "source_broadcast",
            "index_strided",
            "mixed_pairs",
        ],
    )
    def test_scatter_add_pair_single_guard_dispatch(
        self, device, dtype, index_dtype, case
    ):
        source_offset = 1 if case == "source_offset1" else 2 if case == "offset2" else 0
        output_offset = 1 if case == "output_offset1" else 2 if case == "offset2" else 0
        source_storage = torch.full((88 + 4,), 17.0, device=device, dtype=dtype)
        output_storage = torch.full((56 + 4,), 9.0, device=device, dtype=dtype)
        source = (
            source_storage[source_offset : source_offset + 88]
            .reshape(11, 8)
            .fill_(0.125)
        )
        output = (
            output_storage[output_offset : output_offset + 56]
            .reshape(7, 8)
            .fill_(0.125)
        )
        if case == "output_transpose":
            output = torch.full((8, 7), 0.125, device=device, dtype=dtype).t()
        elif case == "source_transpose":
            source = torch.full((8, 11), 0.125, device=device, dtype=dtype).t()
        elif case == "output_strided":
            output = torch.full((7, 16), 0.125, device=device, dtype=dtype)[:, ::2]
        elif case == "source_strided":
            source = torch.full((11, 16), 0.125, device=device, dtype=dtype)[:, ::2]
        elif case == "source_broadcast":
            source = torch.full((1, 8), 0.125, device=device, dtype=dtype).expand(11, 8)
        index = (
            (torch.arange(11, device=device, dtype=index_dtype) * 3 % 7)
            .reshape(11, 1)
            .expand(11, 8)
        )
        if case in ("index_strided", "mixed_pairs"):
            index = index.contiguous()
        if case == "index_strided":
            index = index.repeat_interleave(2, dim=-1)[:, ::2]
        elif case == "mixed_pairs":
            # A single launch contains both same-row and different-row destination pairs.
            delta = torch.tensor(
                [0, 0, 0, 1, 0, 0, 0, 1], device=device, dtype=index_dtype
            )
            index = (index + delta) % 7
            self.assertTrue((index[:, 0::2] == index[:, 1::2]).any().item())
            self.assertTrue((index[:, 0::2] != index[:, 1::2]).any().item())
        expected = (
            output.float()
            .cpu()
            .scatter_add(0, index.long().cpu(), source.float().cpu())
            .to(dtype)
        )
        source_before, index_before = source.clone(), index.clone()
        kernels = self._profile_scatter_add_pair(output, index, source)
        eligible = bool(
            getattr(torch.version, "ppu", None)
        ) and torch.cuda.get_device_capability(device) == (8, 9)
        hit = (
            eligible
            and _SCATTER_ADD_OPT_ENABLED
            and case in ("base", "offset2", "mixed_pairs")
        )
        self._assert_scatter_add_pair_route(kernels, hit, index_dtype)
        self.assertEqual(output.cpu().view(torch.int16), expected.view(torch.int16))
        self.assertEqual(source.view(torch.int16), source_before.view(torch.int16))
        self.assertEqual(index, index_before)

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @serialTest()
    @largeTensorTest("10GB")
    def test_scatter_add_pair_source_64bit(self, device, dtype, index_dtype):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        width = (1 << 30) + 2
        source = torch.full((2, width), 0.125, device=device, dtype=dtype)
        output = torch.zeros((1, width), device=device, dtype=dtype)
        index = torch.zeros((2, 1), device=device, dtype=index_dtype).expand_as(source)
        self.assertGreater(source.numel(), torch.iinfo(torch.int32).max)
        self.assertLess(output.numel(), torch.iinfo(torch.int32).max)
        # Full-shape scatter indices necessarily share the source's large logical numel.
        kernels = self._profile_scatter_add_pair(output, index, source)
        self._assert_scatter_add_pair_route(
            kernels,
            _SCATTER_ADD_OPT_ENABLED,
            index_dtype,
            r"(?:unsigned long(?: long)?|uint64_t)",
        )
        self.assertEqual(output.amin().item(), 0.25)
        self.assertEqual(output.amax().item(), 0.25)
        self.assertEqual(source.amin().item(), 0.125)
        self.assertEqual(source.amax().item(), 0.125)
        self.assertEqual(index[0, 0].item(), 0)
        self.assertEqual(index[-1, -1].item(), 0)

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @serialTest()
    @largeTensorTest("20GB")
    def test_scatter_add_pair_index_stride_64bit(self, device, dtype, index_dtype):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        span = (1 << 31) + 1
        storage = torch.empty(span + 1, device=device, dtype=index_dtype)
        rows = storage.as_strided((2,), (span,))
        rows.copy_(torch.tensor([0, 1], device=device, dtype=index_dtype))
        source = torch.full((2, 8), 0.125, device=device, dtype=dtype)
        output = torch.zeros_like(source)
        index = rows.reshape(2, 1).expand_as(source)
        self.assertEqual(index.stride(), (span, 0))
        kernels = self._profile_scatter_add_pair(output, index, source)
        self._assert_scatter_add_pair_route(
            kernels,
            _SCATTER_ADD_OPT_ENABLED,
            index_dtype,
            r"(?:unsigned long(?: long)?|uint64_t)",
        )
        self.assertEqual(output, source)
        self.assertEqual(rows.cpu(), torch.tensor([0, 1], dtype=index_dtype))

    def _make_scatter_add_special_case(self, dtype, layout):
        info = torch.finfo(dtype)
        row = [
            -0.0,
            info.tiny / 2,
            info.max,
            float("inf"),
            -float("inf"),
            float("nan"),
            1.0,
            -1.0,
        ]
        source = torch.tensor([row, row], dtype=dtype)
        initial = torch.zeros((2, 8), dtype=dtype)
        initial[:, 6], initial[:, 7] = -1.0, 1.0
        index = torch.arange(2).reshape(2, 1).expand_as(source)
        if layout != "broadcast":
            index = index.contiguous()
        if layout == "dense_mixed":
            mask = torch.tensor([False, False, False, True, False, False, False, True])
            index[:, mask] = 1 - index[:, mask]
        # Each output element receives exactly one contribution, even for mixed pairs.
        expected = initial.float().scatter_add(0, index, source.float()).to(dtype)
        return source, index, initial, expected

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("index_layout", ["broadcast", "dense_same", "dense_mixed"])
    def test_scatter_add_pair_special_values(
        self, device, dtype, index_dtype, index_layout
    ):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        values, ids, initial, expected = self._make_scatter_add_special_case(
            dtype, index_layout
        )
        if index_layout == "broadcast":
            index = ids[:, :1].to(device=device, dtype=index_dtype).expand_as(values)
        else:
            index = ids.to(device=device, dtype=index_dtype)
        source, output = values.to(device), initial.to(device)
        source_before, index_before = source.clone(), index.clone()
        kernels = self._profile_scatter_add_pair(output, index, source)
        self._assert_scatter_add_pair_route(
            kernels, _SCATTER_ADD_OPT_ENABLED, index_dtype
        )
        actual = output.cpu()
        self.assertEqual(actual.isnan(), expected.isnan())
        valid = ~expected.isnan()
        self.assertEqual(
            actual.view(torch.int16)[valid], expected.view(torch.int16)[valid]
        )
        self.assertEqual(source.view(torch.int16), source_before.view(torch.int16))
        self.assertEqual(index, index_before)

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize(
        "index_layout", ["broadcast", "dense_same", "dense_split", "dense_mixed"]
    )
    @parametrize("api", ["inplace", "functional", "out"])
    def test_scatter_add_pair_unselected_negative_zero(
        self, device, dtype, index_dtype, index_layout, api
    ):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        initial = torch.zeros((2, 4), dtype=dtype)
        # These low lanes are never selected, even when their paired high lanes are.
        initial[0, ::2] = -0.0
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
        ids = torch.ones((1, 4), dtype=index_dtype)
        if index_layout == "dense_split":
            ids[0, 1::2] = 0
        elif index_layout == "dense_mixed":
            ids[0, 1] = 0
        expected = initial.clone()
        for column in range(4):
            expected[ids[0, column].item(), column] = values[0, column]
        source, output = values.to(device), initial.to(device)
        index = (
            ids[:, :1].to(device).expand_as(source)
            if index_layout == "broadcast"
            else ids.to(device)
        )
        source_before, index_before = source.clone(), index.clone()
        with DeterministicGuard(False), self._capture_pair_profile(device) as prof:
            if api == "functional":
                output = torch.scatter_add(output, 0, index, source)
            elif api == "out":
                torch.scatter_add(initial.to(device), 0, index, source, out=output)
            else:
                output.scatter_add_(0, index, source)
            torch.cuda.synchronize(device)
        self._assert_scatter_add_pair_route(
            _device_kernels(prof), _SCATTER_ADD_OPT_ENABLED, index_dtype
        )
        self.assertEqual(output.cpu().view(torch.int16), expected.view(torch.int16))
        self.assertEqual(source.view(torch.int16), source_before.view(torch.int16))
        self.assertEqual(index, index_before)

    @onlyCUDA
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize(
        "case", ["empty_index", "empty_inner", "bad_shape", "bad_dtype", "overlap"]
    )
    def test_scatter_add_pair_boundary_contract(self, device, dtype, case):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        output = torch.zeros((2, 8), device=device, dtype=dtype)
        source = torch.full_like(output, 0.125)
        index = torch.arange(2, device=device).reshape(2, 1).expand_as(source)
        error = None
        if case == "empty_index":
            index, source = index[:0], source[:0]
        elif case == "empty_inner":
            output, source, index = output[:, :0], source[:, :0], index[:, :0]
        elif case == "bad_shape":
            source = source[:1]
            error = "Expected index|smaller size|no larger"
        elif case == "bad_dtype":
            source = source.float()
            error = r"Expected self\.dtype to be equal to src\.dtype"
        elif case == "overlap":
            source = output
            error = "single memory location|overlap"
        before = output.clone()
        with DeterministicGuard(False), self._capture_pair_profile(device) as prof:
            if error:
                with self.assertRaisesRegex(RuntimeError, error):
                    output.scatter_add_(0, index, source)
            else:
                output.scatter_add_(0, index, source)
            torch.cuda.synchronize(device)
        kernels = _device_kernels(prof)
        if kernels is not None:
            self.assertFalse(
                any(
                    "ppu_scatter_add_opt::" in name or "ppu_index_add_opt::" in name
                    for name in kernels
                )
            )
        self.assertEqual(output.view(torch.int16), before.view(torch.int16))

    @onlyCUDA
    @dtypes(torch.float16)
    @parametrize("bad", [-1, 2])
    @parametrize("lane", [0, 1])
    def test_scatter_add_pair_invalid_index_subprocess(self, device, dtype, bad, lane):
        if not getattr(torch.version, "ppu", None) or torch.cuda.get_device_capability(
            device
        ) != (8, 9):
            self.skipTest("PPU paired atomics require an SM89 device")
        script = textwrap.dedent(r"""
            import sys
            import torch
            device = sys.argv[1]
            torch.cuda.set_device(device)
            torch.use_deterministic_algorithms(False)
            source = torch.ones((2, 8), device=device, dtype=torch.float16)
            index = torch.arange(2, device=device).reshape(2, 1).expand_as(source).clone()
            index[0, int(sys.argv[3])] = int(sys.argv[2])
            try:
                torch.zeros_like(source).scatter_add_(0, index, source)
                torch.cuda.synchronize(device)
            except RuntimeError as error:
                assert "device-side assert" in str(error) or "out of bounds" in str(error), str(error)
                print("EXPECTED_DEVICE_ASSERT")
                sys.exit(0)
            raise AssertionError("An invalid index must not complete successfully")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script, str(device), str(bad), str(lane)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("EXPECTED_DEVICE_ASSERT", result.stdout)


# Set PPU_ATOMIC_FULL_MATRIX=1 to restore the exhaustive Cartesian products.
for _name in ("test_scatter_add_pairs_nd", "test_scatter_add_pair_fastpath_dispatch"):
    setattr(
        TestPPUScatterAdd,
        _name,
        _representative_matrix(getattr(TestPPUScatterAdd, _name)),
    )

# Numerical and tracing cases are collected independently, including subprocess tests.
for _name in (
    "test_scatter_add_pair_nd_dispatch",
    "test_scatter_add_pairs_nd_64bit_addressing",
    "test_scatter_add_pair_fastpath_dispatch",
    "test_gather_pair_fastpath_backward_dispatch",
    "test_scatter_add_pair_fastpath_64bit_addressing",
    "test_scatter_add_pair_disable_env",
    "test_scatter_add_pair_single_guard_dispatch",
    "test_scatter_add_pair_source_64bit",
    "test_scatter_add_pair_index_stride_64bit",
    "test_scatter_add_pair_special_values",
    "test_scatter_add_pair_boundary_contract",
    "test_scatter_add_pair_unselected_negative_zero",
):
    setattr(
        TestPPUScatterAdd, _name, _with_route_check(getattr(TestPPUScatterAdd, _name))
    )

instantiate_device_type_tests(TestPPUScatterAdd, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
