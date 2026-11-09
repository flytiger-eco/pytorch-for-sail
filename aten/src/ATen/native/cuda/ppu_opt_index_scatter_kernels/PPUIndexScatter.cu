#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/ppu_opt_index_scatter_kernels/PPUIndexScatter.h>

#if defined(USE_PPU)

#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace at::native::ppu_index_scatter_opt {

inline constexpr int kPairAtomicThreads = 256;
inline constexpr int kPairAtomicBlocksPerSM = 8;

template <typename scalar_t>
struct PairAtomicTraits;

template <>
struct PairAtomicTraits<at::Half> {
  using native_type = __half;
  using pair_type = __half2;
};

template <>
struct PairAtomicTraits<at::BFloat16> {
  using native_type = __nv_bfloat16;
  using pair_type = __nv_bfloat162;
};

static_assert(sizeof(__half2) == sizeof(uint32_t));
static_assert(sizeof(__nv_bfloat162) == sizeof(uint32_t));

// Read two valid source values per thread; combine updates only for adjacent destinations.
template <typename scalar_t, typename offset_t, bool Scatter, bool Scale>
__global__ void pair_atomic_add_kernel(
    scalar_t* output,
    const typename PairAtomicTraits<scalar_t>::pair_type* source,
    const int64_t* index,
    offset_t pairs,
    offset_t half_width,
    offset_t output_rows,
    offset_t output_numel,
    offset_t index_row_stride,
    offset_t index_column_stride,
    scalar_t alpha) {
  using native_t = typename PairAtomicTraits<scalar_t>::native_type;
  using pair_t = typename PairAtomicTraits<scalar_t>::pair_type;
  const offset_t grid_stride = static_cast<offset_t>(blockDim.x) * gridDim.x;
  for (offset_t q = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       q < pairs; q += grid_stride) {
    const offset_t row = q / half_width;
    const offset_t column = (q % half_width) * 2;
    const offset_t index_offset =
        row * index_row_stride + (Scatter ? column * index_column_stride : 0);
    const int64_t dst0 = index[index_offset];
    const int64_t dst1 = Scatter && index_column_stride != 0
        ? index[index_offset + index_column_stride]
        : dst0;
    CUDA_KERNEL_ASSERT(dst0 >= 0 && dst0 < static_cast<int64_t>(output_rows));
    CUDA_KERNEL_ASSERT(dst1 >= 0 && dst1 < static_cast<int64_t>(output_rows));
    auto value = source[q];
    if constexpr (Scale) {
      // Match the original index_add multiplication: multiply in FP32, then round each product to the tensor dtype.
      value.x = static_cast<native_t>(scalar_t(value.x) * alpha);
      value.y = static_cast<native_t>(scalar_t(value.y) * alpha);
    }
    const offset_t offset0 =
        static_cast<offset_t>(dst0) * (half_width * 2) + column;
    const offset_t offset1 =
        static_cast<offset_t>(dst1) * (half_width * 2) + column + 1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
    if (!Scatter || dst0 == dst1) {
      // Use the SDK pair overload without adding assembly or changing SDK gates.
      atomicAdd(reinterpret_cast<pair_t*>(output + offset0), value);
      continue;
    }
#endif
    fastAtomicAdd(output, offset0, output_numel, scalar_t(value.x), true);
    fastAtomicAdd(output, offset1, output_numel, scalar_t(value.y), true);
  }
}

template <typename scalar_t, typename offset_t, bool Scatter, bool Scale = false>
static void launch_pair_atomic_add(
    const Tensor& output,
    const Tensor& source,
    const Tensor& index,
    offset_t pairs,
    offset_t half_width,
    int64_t index_row_stride,
    int64_t index_column_stride,
    int blocks,
    scalar_t alpha = scalar_t(1.0f)) {
  using pair_t = typename PairAtomicTraits<scalar_t>::pair_type;
  const auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  pair_atomic_add_kernel<scalar_t, offset_t, Scatter, Scale>
      <<<blocks, kPairAtomicThreads, 0, stream>>>(
      output.mutable_data_ptr<scalar_t>(),
      reinterpret_cast<const pair_t*>(source.const_data_ptr<scalar_t>()),
      index.const_data_ptr<int64_t>(),
      pairs,
      half_width,
      static_cast<offset_t>(output.size(0)),
      static_cast<offset_t>(output.numel()),
      static_cast<offset_t>(index_row_stride),
      static_cast<offset_t>(index_column_stride),
      alpha);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename offset_t, bool Scatter>
static void dispatch_pair_atomic_add(
    const Tensor& output,
    const Tensor& source,
    const Tensor& index,
    int blocks,
    const Scalar& alpha) {
  const auto pairs = static_cast<offset_t>(source.numel() / 2);
  const auto half_width = static_cast<offset_t>(source.size(1) / 2);
  if constexpr (Scatter) {
    if (index.stride(0) == 1 && index.stride(1) == 0) {
      // Read broadcast indices by row without copying indices or dropping source contributions.
      launch_pair_atomic_add<scalar_t, offset_t, false>(
          output, source, index, pairs, half_width, 1, 0, blocks);
    } else {
      launch_pair_atomic_add<scalar_t, offset_t, true>(
          output, source, index, pairs, half_width,
          index.stride(0), index.stride(1), blocks);
    }
  } else {
    if (!alpha.isSymbolic() && alpha.equal(1)) {
      launch_pair_atomic_add<scalar_t, offset_t, false, false>(
          output, source, index, pairs, half_width, index.stride(0), 0, blocks);
    } else {
      // Convert alpha as in the original path; rounding to 1 does not select the multiplication-free path.
      launch_pair_atomic_add<scalar_t, offset_t, false, true>(
          output, source, index, pairs, half_width, index.stride(0), 0, blocks,
          alpha.to<scalar_t>());
    }
  }
}

// Called only after native CUDA index_add/scatter_add meta validation and deterministic routing.
// The dispatcher handles negative bits, wrappers check devices, and meta validates dtype and shape consistency.
template <bool Scatter>
static bool try_launch_pair_atomic_add(
    const Tensor& output,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha = 1) {
  if (dim != 0 ||
      (output.scalar_type() != at::kHalf && output.scalar_type() != at::kBFloat16) ||
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
  if (reinterpret_cast<std::uintptr_t>(output.const_data_ptr()) %
              sizeof(uint32_t) != 0 ||
      reinterpret_cast<std::uintptr_t>(source.const_data_ptr()) %
              sizeof(uint32_t) != 0 ||
      get_overlap_status(output, source) != MemOverlapStatus::No ||
      output.storage().is_alias_of(index.storage())) {
    return false;
  }

  // Query cached properties of the current device rather than caching the first device's eligibility.
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major != 8 || prop->minor != 9) {
    return false;
  }
  // Numel is int64_t and pairs is at most INT64_MAX / 2, so rounding up is safe.
  const int64_t pairs = source.numel() / 2;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (pairs + kPairAtomicThreads - 1) / kPairAtomicThreads,
      std::min<int64_t>(
          static_cast<int64_t>(prop->multiProcessorCount) * kPairAtomicBlocksPerSM,
          prop->maxGridSize[0])));
  // Select the addressing width without excluding tensors that need 64-bit offsets.
  // Index values remain int64_t; all three tensors use the selected offset width.
  const bool use_32bit =
      canUse32BitIndexMath(output) && canUse32BitIndexMath(source) &&
      canUse32BitIndexMath(index);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(output.scalar_type(), "ppu_pair_atomic_add", [&] {
    if (use_32bit) {
      dispatch_pair_atomic_add<scalar_t, uint32_t, Scatter>(
          output, source, index, blocks, alpha);
    } else {
      dispatch_pair_atomic_add<scalar_t, uint64_t, Scatter>(
          output, source, index, blocks, alpha);
    }
  });
  return true;
}

bool try_launch_index_add_opt(
    const Tensor& output,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha) {
  return try_launch_pair_atomic_add<false>(output, dim, index, source, alpha);
}

bool try_launch_scatter_add_opt(
    const Tensor& output,
    int64_t dim,
    const Tensor& index,
    const Tensor& source) {
  return try_launch_pair_atomic_add<true>(output, dim, index, source);
}

} // namespace at::native::ppu_index_scatter_opt

#endif
