#pragma once

#if defined(USE_PPU)

#include <ATen/MemoryOverlap.h>
#include <ATen/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace at::native::ppu {

inline constexpr int kBF16PairThreads = 256;
inline constexpr int kBF16PairBlocksPerSM = 8;

// Read two valid source values per thread; combine updates only for adjacent destinations.
template <bool Scatter, bool Scale>
__global__ void bf16_pair_atomic_add_kernel(
    at::BFloat16* output,
    const __nv_bfloat162* source,
    const int64_t* index,
    uint32_t pairs,
    uint32_t half_width,
    uint32_t output_rows,
    uint32_t output_numel,
    int64_t index_row_stride,
    int64_t index_column_stride,
    at::BFloat16 alpha) {
  for (uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < pairs;
       q += blockDim.x * gridDim.x) {
    const uint32_t row = q / half_width;
    const uint32_t column = (q % half_width) * 2;
    const int64_t index_offset =
        row * index_row_stride + (Scatter ? column * index_column_stride : 0);
    const int64_t dst0 = index[index_offset];
    const int64_t dst1 = Scatter && index_column_stride != 0
        ? index[index_offset + index_column_stride]
        : dst0;
    CUDA_KERNEL_ASSERT(dst0 >= 0 && dst0 < output_rows);
    CUDA_KERNEL_ASSERT(dst1 >= 0 && dst1 < output_rows);
    auto value = source[q];
    if constexpr (Scale) {
      // Match the original index_add BF16 multiplication: multiply in FP32, then round each product to BF16.
      value.x = static_cast<__nv_bfloat16>(at::BFloat16(value.x) * alpha);
      value.y = static_cast<__nv_bfloat16>(at::BFloat16(value.y) * alpha);
    }
    const uint32_t offset0 =
        static_cast<uint32_t>(dst0) * (half_width * 2) + column;
    const uint32_t offset1 =
        static_cast<uint32_t>(dst1) * (half_width * 2) + column + 1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (!Scatter || dst0 == dst1) {
      // Use the SDK BF16x2 overload without adding assembly or changing SDK gates.
      atomicAdd(reinterpret_cast<__nv_bfloat162*>(output + offset0), value);
      continue;
    }
#endif
    fastAtomicAdd(output, offset0, output_numel, at::BFloat16(value.x), true);
    fastAtomicAdd(output, offset1, output_numel, at::BFloat16(value.y), true);
  }
}

template <bool Scatter, bool Scale = false>
static void launch_bf16_pair_atomic_add(
    const Tensor& output,
    const Tensor& source,
    const Tensor& index,
    uint32_t pairs,
    uint32_t half_width,
    int64_t index_row_stride,
    int64_t index_column_stride,
    int blocks,
    at::BFloat16 alpha = at::BFloat16(1.0f)) {
  const auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  bf16_pair_atomic_add_kernel<Scatter, Scale>
      <<<blocks, kBF16PairThreads, 0, stream>>>(
      output.mutable_data_ptr<at::BFloat16>(),
      reinterpret_cast<const __nv_bfloat162*>(
          source.const_data_ptr<at::BFloat16>()),
      index.const_data_ptr<int64_t>(),
      pairs,
      half_width,
      static_cast<uint32_t>(output.size(0)),
      static_cast<uint32_t>(output.numel()),
      index_row_stride,
      index_column_stride,
      alpha);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Called only after native CUDA index_add/scatter_add meta validation and deterministic routing.
// The dispatcher handles negative bits, wrappers check devices, and meta validates dtype and shape consistency.
template <bool Scatter>
static bool try_launch_bf16_pair_atomic_add(
    const Tensor& output,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha = 1) {
  if (dim != 0 || output.scalar_type() != at::kBFloat16 ||
      index.scalar_type() != at::kLong || output.layout() != at::kStrided ||
      source.layout() != at::kStrided || index.layout() != at::kStrided ||
      output.dim() != 2) {
    return false;
  }
  if (output.numel() == 0 || source.numel() == 0 || output.size(1) % 2 != 0 ||
      !output.is_contiguous() || !source.is_contiguous()) {
    return false;
  }
  if constexpr (Scatter) {
    // Scatter meta allows partial indices; this kernel requires full source coverage and matching row widths.
    if (index.sizes() != source.sizes() || output.size(1) != source.size(1)) {
      return false;
    }
  } else {
    if (alpha.isComplex() || index.dim() != 1 || !index.is_contiguous()) {
      return false;
    }
  }
  // Source/output addressing uses 32 bits; index strides and storage offsets remain 64-bit.
  if (!canUse32BitIndexMath(output) || !canUse32BitIndexMath(source) ||
      reinterpret_cast<std::uintptr_t>(output.const_data_ptr()) %
              sizeof(__nv_bfloat162) != 0 ||
      reinterpret_cast<std::uintptr_t>(source.const_data_ptr()) %
              sizeof(__nv_bfloat162) != 0 ||
      get_overlap_status(output, source) != MemOverlapStatus::No ||
      output.storage().is_alias_of(index.storage())) {
    return false;
  }

  // Query cached properties of the current device rather than caching the first device's eligibility.
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major != 8 || prop->minor != 9) {
    return false;
  }
  const auto pairs = static_cast<uint32_t>(source.numel() / 2);
  const auto half_width = static_cast<uint32_t>(source.size(1) / 2);
  const int blocks = static_cast<int>(std::min<int64_t>(
      (static_cast<int64_t>(pairs) + kBF16PairThreads - 1) / kBF16PairThreads,
      std::min<int64_t>(
          static_cast<int64_t>(prop->multiProcessorCount) * kBF16PairBlocksPerSM,
          prop->maxGridSize[0])));
  if constexpr (Scatter) {
    if (index.stride(0) == 1 && index.stride(1) == 0) {
      // Read broadcast indices by row without copying indices or dropping source contributions.
      launch_bf16_pair_atomic_add<false>(
          output, source, index, pairs, half_width, 1, 0, blocks);
    } else {
      launch_bf16_pair_atomic_add<true>(
          output, source, index, pairs, half_width,
          index.stride(0), index.stride(1), blocks);
    }
  } else {
    if (alpha.equal(1)) {
      launch_bf16_pair_atomic_add<false>(
          output, source, index, pairs, half_width, index.stride(0), 0, blocks);
    } else {
      // Convert alpha as in the original path; rounding to 1 does not select the multiplication-free path.
      launch_bf16_pair_atomic_add<false, true>(
          output, source, index, pairs, half_width, index.stride(0), 0, blocks,
          alpha.to<at::BFloat16>());
    }
  }
  return true;
}

} // namespace at::native::ppu

#endif
