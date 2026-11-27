import math
import os
import subprocess
import sys

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests


DISABLE_OPT = "PYTORCH_DISABLE_PPU_CHUNK_CAT_OPT"


def chunk_cat_reference(tensors, dim, num_chunks, dtype):
    dim = dim % tensors[0].ndim
    pieces = []
    for tensor in tensors:
        rows = (tensor.size(dim) + num_chunks - 1) // num_chunks
        shape = list(tensor.shape)
        shape[dim] = rows * num_chunks
        padded = torch.zeros(shape, dtype=dtype, device="cpu")
        padded.narrow(dim, 0, tensor.size(dim)).copy_(tensor.cpu())
        pieces.append(padded.reshape(*shape[:dim], num_chunks, -1))
    return torch.cat(pieces, dim=dim + 1)


class TestChunkCat(TestCase):
    def check_chunk_cat(self, tensors, dim, chunks, out):
        expected = chunk_cat_reference(tensors, dim, chunks, out.dtype)
        result = torch.ops.aten._chunk_cat.out(tensors, dim, chunks, out=out)
        self.assertIs(result, out)
        self.assertEqual(out, expected.to(out.device), atol=0, rtol=0)
        return expected

    def inputs(self, shapes, device, offset=0, dtype=torch.bfloat16):
        tensors = []
        for i, shape in enumerate(shapes):
            count = math.prod(shape)
            storage = torch.arange(count + offset, dtype=torch.float32)
            storage = (storage.remainder(251) - 125 + i).to(dtype).to(device)
            tensors.append(storage[offset:].reshape(shape))
        return tensors

    @parametrize("chunks", [1, 2, 4, 8])
    def test_mixed_shapes(self, device, chunks):
        tensors = self.inputs(
            [(128, 64), (64, 128), (12,), (16,), (128,), (9, 1, 4)], device
        )
        out = torch.empty(0, dtype=torch.float32, device=device)
        self.check_chunk_cat(tensors, 0, chunks, out)

    @parametrize("shape", [(1,), (7,), (8,), (9,), (1023,), (1024,), (1025,), (1033, 7)])
    @parametrize("chunks", [1, 2, 10])
    @parametrize("offset", [0, 1])
    def test_alignment_tail_padding(self, device, shape, chunks, offset):
        tensors = self.inputs([shape, (3,)], device, offset=offset)
        expected = chunk_cat_reference(tensors, 0, chunks, torch.float32)
        storage = torch.full(
            (expected.numel() + offset + 8,), -42.0, device=device
        )
        out = storage[offset:offset + expected.numel()].view(expected.shape)
        self.check_chunk_cat(tensors, 0, chunks, out)
        self.assertEqual(storage[:offset], torch.full_like(storage[:offset], -42))
        self.assertEqual(storage[-8:], torch.full_like(storage[-8:], -42))

    @parametrize("dim", [0, 1, -1, -2])
    def test_dimensions(self, device, dim):
        tensors = self.inputs([(3, 11), (3, 7)], device)
        if dim in (0, -2):
            tensors[1] = self.inputs([(5, 7)], device)[0]
        self.check_chunk_cat(
            tensors, dim, 4, torch.empty(0, dtype=torch.float32, device=device)
        )

    @parametrize("layout", ["strided_input", "expanded_input", "strided_output"])
    def test_layout_fallback(self, device, layout):
        tensors = self.inputs([(33, 16), (12,)], device)
        if layout == "strided_input":
            tensors[0] = tensors[0][:, ::2]
        elif layout == "expanded_input":
            tensors[0] = tensors[0][:1].expand(33, 16)
        expected = chunk_cat_reference(tensors, 0, 4, torch.float32)
        if layout == "strided_output":
            storage = torch.full(
                (expected.shape[0], expected.shape[1] * 2), -42.0, device=device
            )
            out = storage[:, ::2]
        else:
            out = torch.empty_like(expected, device=device)
        self.check_chunk_cat(tensors, 0, 4, out)
        if layout == "strided_output":
            self.assertEqual(storage[:, 1::2], torch.full_like(storage[:, 1::2], -42))

    @parametrize("input_dtype", [torch.float32, torch.bfloat16])
    @parametrize("output_dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_dtype_fallback(self, device, input_dtype, output_dtype):
        tensors = self.inputs([(17, 16), (7,)], device, dtype=input_dtype)
        self.check_chunk_cat(
            tensors, 0, 4, torch.empty(0, dtype=output_dtype, device=device)
        )

    def test_all_bfloat16_bits(self, device):
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        tensor = bits.view(torch.bfloat16).to(device)
        out = torch.empty((1, 65536), dtype=torch.float32, device=device)
        torch.ops.aten._chunk_cat.out([tensor], 0, 1, out=out)
        expected = tensor.float()
        finite = ~torch.isnan(expected)
        self.assertEqual(out.view(-1)[finite], expected[finite], atol=0, rtol=0)
        self.assertEqual(
            out.view(-1)[finite].view(torch.int32), expected[finite].view(torch.int32)
        )
        self.assertEqual(torch.isnan(out.view(-1)), torch.isnan(expected))

    @onlyCUDA
    def test_nondefault_stream(self, device):
        tensors = self.inputs([(1025, 16), (12,)], device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        out = torch.empty(0, dtype=torch.float32, device=device)
        with torch.cuda.stream(stream):
            for i in range(8):
                for tensor in tensors:
                    tensor.fill_(i)
                torch.ops.aten._chunk_cat.out(tensors, 0, 4, out=out)
        stream.synchronize()
        expected = chunk_cat_reference(tensors, 0, 4, out.dtype)
        self.assertEqual(out, expected.to(device), atol=0, rtol=0)

    @parametrize("invalid", ["empty_list", "empty_tensor", "chunks", "mixed_dtype"])
    def test_validation(self, device, invalid):
        tensors = self.inputs([(8,)], device)
        chunks = 2
        if invalid == "empty_list":
            tensors = []
        elif invalid == "empty_tensor":
            tensors = [torch.empty(0, dtype=torch.bfloat16, device=device)]
        elif invalid == "chunks":
            chunks = 0
        else:
            tensors.append(tensors[0].float())
        with self.assertRaises(RuntimeError):
            torch.ops.aten._chunk_cat.out(
                tensors, 0, chunks, out=torch.empty(0, device=device)
            )

    @onlyCUDA
    @parametrize("disabled", [False, True])
    def test_routing(self, device, disabled):
        if not getattr(torch.version, "ppu", None):
            self.skipTest("需要 USE_PPU 构建")
        prop = torch.cuda.get_device_properties(device)
        if (prop.major, prop.minor) != (8, 9):
            self.skipTest("需要 PPU 890")
        script = r'''
import torch
fast = "ppu_chunk_cat_bf16_float_vector_kernel"
def run(inputs, dim, out, expected_fast):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        torch.ops.aten._chunk_cat.out(inputs, dim, 4, out=out)
        torch.cuda.synchronize()
    hit = any(fast in e.name for e in p.events())
    assert hit == expected_fast, (hit, expected_fast, [e.name for e in p.events()])
x = torch.arange(1025 * 16, device="cuda", dtype=torch.float32).to(torch.bfloat16).view(1025, 16)
y = torch.ones(12, device="cuda", dtype=torch.bfloat16)
expected = EXPECTED_FAST
run([x, y], 0, torch.empty(0, device="cuda"), expected)
run([x.view(-1)[1:], y], 0, torch.empty(0, device="cuda"), expected)
run([x], -2, torch.empty(0, device="cuda"), expected)
run([x[:, ::2], y], 0, torch.empty(0, device="cuda"), False)
run([x], 1, torch.empty(0, device="cuda"), False)
run([x.float()], 0, torch.empty(0, device="cuda"), False)
run([x], 0, torch.empty(0, device="cuda", dtype=torch.bfloat16), False)
run([x, y], 0, torch.empty((4, 8230), device="cuda")[:, ::2], False)
print("ROUTING_PASS")
'''.replace("EXPECTED_FAST", repr(not disabled))
        env = os.environ.copy()
        env[DISABLE_OPT] = "1" if disabled else "0"
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        visible = env.get("CUDA_VISIBLE_DEVICES")
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index] if visible else str(index)
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True,
            text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("ROUTING_PASS", result.stdout)


instantiate_device_type_tests(TestChunkCat, globals())

if __name__ == "__main__":
    run_tests()
