#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/ppu_opt_chunk_cat_kernels/PPUChunkCat.h>

#if defined(USE_PPU)

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/util/TypeCast.h>
#include <c10/util/env.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace at::native::ppu_chunk_cat_opt {

constexpr int kThreads = 128;
constexpr int kElementsPerThread = 8;
constexpr int64_t kElementsPerBlock = kThreads * kElementsPerThread;
constexpr int kMetadataFields = 5;

__global__ void ppu_chunk_cat_bf16_float_vector_kernel(
    const int64_t* metadata,
    const int64_t* block_to_tensor,
    float* output,
    int64_t output_chunk_size) {
  const int64_t tensor_idx = block_to_tensor[blockIdx.x];
  const int64_t* info = metadata + tensor_idx * kMetadataFields;
  const int64_t input_size = info[1];
  const int64_t chunk_size = info[2];
  const int64_t chunk_idx = blockIdx.y;
  const int64_t tile_idx = static_cast<int64_t>(blockIdx.x) - info[4];
  const int64_t offset =
      (tile_idx * kThreads + threadIdx.x) * kElementsPerThread;
  if (offset >= chunk_size) {
    return;
  }

  const int64_t input_start = chunk_idx * chunk_size;
  const int64_t valid_size =
      std::min(chunk_size, std::max(int64_t{0}, input_size - input_start));
  const auto* input = reinterpret_cast<const BFloat16*>(info[0]);
  float* dst = output + chunk_idx * output_chunk_size + info[3] + offset;
  float values[kElementsPerThread] = {};
  using InputVector = memory::aligned_vector<BFloat16, kElementsPerThread>;
  using OutputVector = memory::aligned_vector<float, 4>;

  if (offset + kElementsPerThread <= valid_size &&
      reinterpret_cast<std::uintptr_t>(input + input_start + offset) %
              alignof(InputVector) ==
          0) {
    const auto value = *reinterpret_cast<const InputVector*>(
        input + input_start + offset);
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
      values[i] = static_cast_with_inter_type<float, BFloat16>::apply(value.val[i]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
      if (offset + i < valid_size) {
        values[i] = static_cast_with_inter_type<float, BFloat16>::apply(
            input[input_start + offset + i]);
      }
    }
  }

  if (offset + kElementsPerThread <= chunk_size &&
      reinterpret_cast<std::uintptr_t>(dst) % alignof(OutputVector) == 0) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      OutputVector value;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        value.val[i] = values[j * 4 + i];
      }
      *reinterpret_cast<OutputVector*>(dst + j * 4) = value;
    }
  } else {
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
      if (offset + i < chunk_size) {
        dst[i] = values[i];
      }
    }
  }
}

bool try_launch_chunk_cat_opt(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out) {
  static const bool enabled =
      !c10::utils::check_env("PYTORCH_DISABLE_PPU_CHUNK_CAT_OPT").value_or(false);
  if (!enabled || dim != 0 || tensors.empty() || num_chunks < 1 ||
      out.scalar_type() != at::kFloat || out.layout() != at::kStrided ||
      !out.is_cuda() || !out.is_contiguous() || out.is_neg() || out.is_conj()) {
    return false;
  }
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major != 8 || prop->minor != 9 ||
      num_chunks > prop->maxGridSize[1]) {
    return false;
  }
  for (const auto& tensor : tensors) {
    if (tensor.scalar_type() != at::kBFloat16 ||
        tensor.layout() != at::kStrided || tensor.device() != out.device() ||
        tensor.dim() == 0 || tensor.numel() == 0 || !tensor.is_contiguous() ||
        tensor.is_neg() || tensor.is_conj() ||
        out.storage().is_alias_of(tensor.storage())) {
      return false;
    }
  }

  std::vector<int64_t> metadata;
  std::vector<int64_t> block_to_tensor;
  metadata.reserve(tensors.size() * kMetadataFields);
  int64_t output_chunk_size = 0;
  int64_t num_blocks = 0;
  constexpr auto max_size = std::numeric_limits<int64_t>::max();
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    const int64_t trailing_size = tensor.numel() / tensor.size(0);
    const int64_t rows_per_chunk =
        tensor.size(0) / num_chunks + (tensor.size(0) % num_chunks != 0);
    const int64_t chunk_size = rows_per_chunk * trailing_size;
    const int64_t tensor_blocks =
        chunk_size / kElementsPerBlock + (chunk_size % kElementsPerBlock != 0);
    if (chunk_size > max_size / num_chunks ||
        output_chunk_size > max_size / num_chunks - chunk_size ||
        tensor_blocks > static_cast<int64_t>(prop->maxGridSize[0]) - num_blocks) {
      return false;
    }
    metadata.insert(
        metadata.end(),
        {reinterpret_cast<int64_t>(tensor.const_data_ptr<BFloat16>()),
         tensor.numel(),
         chunk_size,
         output_chunk_size,
         num_blocks});
    block_to_tensor.insert(block_to_tensor.end(), tensor_blocks, i);
    num_blocks += tensor_blocks;
    output_chunk_size += chunk_size;
  }
  if (output_chunk_size * num_chunks > max_size / sizeof(float)) {
    return false;
  }

  const int64_t metadata_size = static_cast<int64_t>(metadata.size());
  auto packed_cpu = at::empty(
      {metadata_size + num_blocks},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  auto* packed_data = packed_cpu.mutable_data_ptr<int64_t>();
  std::memcpy(packed_data, metadata.data(), metadata_size * sizeof(int64_t));
  std::memcpy(
      packed_data + metadata_size,
      block_to_tensor.data(),
      num_blocks * sizeof(int64_t));
  auto packed = packed_cpu.to(out.device(), /*non_blocking=*/true);
  at::native::resize_output(out, {num_chunks, output_chunk_size});
  const auto* device_metadata = packed.const_data_ptr<int64_t>();
  ppu_chunk_cat_bf16_float_vector_kernel<<<
      dim3(num_blocks, num_chunks),
      kThreads,
      0,
      at::cuda::getCurrentCUDAStream(out.get_device())>>>(
      device_metadata,
      device_metadata + metadata_size,
      out.mutable_data_ptr<float>(),
      output_chunk_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

}

#endif
