#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/ppu_opt_scatter_add_kernels/PPUScatterAdd.h>

#if defined(USE_PPU)

#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/NonEmptyUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace at::native::ppu_scatter_add_opt {

inline constexpr int kScatterAddThreads = 256;
inline constexpr int kScatterAddBlocksPerSM = 8;

enum class ScatterAddPath {
  Native,
  Broadcast,
  Dense,
};

// Keep performance admission separate from the public operator's supported layouts.
// New structural fast paths belong here; unmatched structures must use native routing.
static ScatterAddPath select_scatter_add_path(
    bool contiguous,
    int64_t dim,
    int64_t inner,
    bool aligned,
    bool broadcast,
    bool full_dense) {
  if (contiguous && dim == 0 && inner % 2 == 0 && aligned) {
    if (broadcast) {
      return ScatterAddPath::Broadcast;
    }
    if (full_dense) {
      // Dense indices may target different rows; the kernel checks each pair before combining.
      return ScatterAddPath::Dense;
    }
  }
  return ScatterAddPath::Native;
}

template <typename scalar_t>
struct ScatterAddAtomicTraits;

template <>
struct ScatterAddAtomicTraits<at::Half> {
  using native_type = __half;
  using pair_type = __half2;
};

template <>
struct ScatterAddAtomicTraits<at::BFloat16> {
  using native_type = __nv_bfloat16;
  using pair_type = __nv_bfloat162;
};

static_assert(sizeof(__half2) == sizeof(uint32_t));
static_assert(sizeof(__nv_bfloat162) == sizeof(uint32_t));

// Broadcast indices share a row; dense indices select each destination independently.
template <typename scalar_t, typename offset_t, typename index_t, bool Broadcast>
__global__ void scatter_add_pair_kernel(
    scalar_t* output,
    const typename ScatterAddAtomicTraits<scalar_t>::pair_type* source,
    const index_t* index,
    offset_t pairs,
    offset_t half_width,
    offset_t output_rows,
    offset_t output_numel,
    offset_t index_row_stride,
    offset_t index_column_stride) {
  using pair_t = typename ScatterAddAtomicTraits<scalar_t>::pair_type;
  const offset_t grid_stride = static_cast<offset_t>(blockDim.x) * gridDim.x;
  for (offset_t q = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       q < pairs; q += grid_stride) {
    const offset_t row = q / half_width;
    const offset_t column = (q % half_width) * 2;
    const offset_t index_offset =
        row * index_row_stride + (Broadcast ? 0 : column * index_column_stride);
    const int64_t dst0 = index[index_offset];
    const int64_t dst1 = !Broadcast && index_column_stride != 0
        ? index[index_offset + index_column_stride]
        : dst0;
    CUDA_KERNEL_ASSERT(dst0 >= 0 && dst0 < static_cast<int64_t>(output_rows));
    CUDA_KERNEL_ASSERT(dst1 >= 0 && dst1 < static_cast<int64_t>(output_rows));
    auto value = source[q];
    const offset_t offset0 =
        static_cast<offset_t>(dst0) * (half_width * 2) + column;
    const offset_t offset1 =
        static_cast<offset_t>(dst1) * (half_width * 2) + column + 1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
    if (Broadcast || dst0 == dst1) {
      // Use the SDK pair overload without adding assembly or changing SDK gates.
      atomicAdd(reinterpret_cast<pair_t*>(output + offset0), value);
      continue;
    }
#endif
    fastAtomicAdd(output, offset0, output_numel, scalar_t(value.x), true);
    fastAtomicAdd(output, offset1, output_numel, scalar_t(value.y), true);
  }
}

template <typename scalar_t, typename offset_t, typename index_t, bool Broadcast>
static void launch_scatter_add_pair(
    const Tensor& output,
    const Tensor& source,
    const Tensor& index,
    offset_t pairs,
    offset_t half_width,
    int64_t index_row_stride,
    int64_t index_column_stride,
    int blocks) {
  using pair_t = typename ScatterAddAtomicTraits<scalar_t>::pair_type;
  const auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  scatter_add_pair_kernel<scalar_t, offset_t, index_t, Broadcast>
      <<<blocks, kScatterAddThreads, 0, stream>>>(
      output.mutable_data_ptr<scalar_t>(),
      reinterpret_cast<const pair_t*>(source.const_data_ptr<scalar_t>()),
      index.const_data_ptr<index_t>(),
      pairs,
      half_width,
      static_cast<offset_t>(output.size(0)),
      static_cast<offset_t>(output.numel()),
      static_cast<offset_t>(index_row_stride),
      static_cast<offset_t>(index_column_stride));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename offset_t, typename index_t>
static void dispatch_scatter_add(
    const Tensor& output,
    int64_t dim,
    const Tensor& source,
    const Tensor& index,
    int64_t inner,
    ScatterAddPath path,
    int blocks) {
  const auto pairs = static_cast<offset_t>(source.numel() / 2);
  const auto half_width = static_cast<offset_t>(inner / 2);
  if (path == ScatterAddPath::Broadcast) {
    launch_scatter_add_pair<scalar_t, offset_t, index_t, true>(
        output, source, index, pairs, half_width, ensure_nonempty_stride(index, dim), 0, blocks);
  } else {
    launch_scatter_add_pair<scalar_t, offset_t, index_t, false>(
        output, source, index, pairs, half_width, inner, 1, blocks);
  }
}

// Called only after native CUDA scatter_add meta validation and deterministic routing.
// The dispatcher handles negative bits, wrappers check devices, and meta validates dtype and shape consistency.
bool try_launch_scatter_add_opt(
    const Tensor& output,
    int64_t dim,
    const Tensor& index,
    const Tensor& source) {
  const bool index_supported = index.scalar_type() == at::kLong ||
      index.scalar_type() == at::kInt;
  if ((output.scalar_type() != at::kHalf && output.scalar_type() != at::kBFloat16) ||
      !index_supported || output.layout() != at::kStrided ||
      source.layout() != at::kStrided || index.layout() != at::kStrided) {
    return false;
  }
  if (output.numel() == 0 || source.numel() == 0 || index.numel() == 0) {
    return false;
  }
  const bool contiguous = output.is_contiguous() && source.is_contiguous();
  if (!contiguous) {
    return false;
  }
  // Overlap analysis may report TooHard for slices or expanded sources in distinct storages.
  if ((output.storage().is_alias_of(source.storage()) &&
       get_overlap_status(output, source) != MemOverlapStatus::No) ||
      output.storage().is_alias_of(index.storage())) {
    return false;
  }

  // Query cached properties of the current device rather than caching the first device's eligibility.
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major != 8 || prop->minor != 9) {
    return false;
  }
  int64_t inner = 1;
  for (int64_t d = dim + 1; d < source.dim(); ++d) {
    inner *= source.size(d);
  }
  bool full_shape = index.sizes() == source.sizes();
  for (int64_t d = 0; d < output.dim(); ++d) {
    if (d != dim && output.size(d) != source.size(d)) {
      full_shape = false;
    }
  }
  bool broadcast = full_shape;
  for (int64_t d = 0; d < index.dim(); ++d) {
    if (d != dim && index.size(d) > 1 && index.stride(d) != 0) {
      broadcast = false;
    }
  }
  const bool aligned =
      reinterpret_cast<std::uintptr_t>(output.const_data_ptr()) % sizeof(uint32_t) == 0 &&
      reinterpret_cast<std::uintptr_t>(source.const_data_ptr()) % sizeof(uint32_t) == 0;
  const bool full_dense = full_shape && index.is_contiguous();
  const auto path = select_scatter_add_path(contiguous, dim, inner, aligned, broadcast, full_dense);
  if (path == ScatterAddPath::Native) {
    return false;
  }
  const int64_t numel = source.numel();
  const int64_t pairs = numel / 2 + numel % 2;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (pairs + kScatterAddThreads - 1) / kScatterAddThreads,
      std::min<int64_t>(
          static_cast<int64_t>(prop->multiProcessorCount) * kScatterAddBlocksPerSM,
          prop->maxGridSize[0])));
  // Index values retain their storage dtype; all three tensors use the selected offset width.
  const bool use_32bit =
      canUse32BitIndexMath(output) && canUse32BitIndexMath(source) &&
      canUse32BitIndexMath(index);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(output.scalar_type(), "ppu_scatter_add_pair", [&] {
    const auto launch = [&](auto index_type) {
      using index_t = decltype(index_type);
      if (use_32bit) {
        dispatch_scatter_add<scalar_t, uint32_t, index_t>(
            output, dim, source, index, inner, path, blocks);
      } else {
        dispatch_scatter_add<scalar_t, uint64_t, index_t>(
            output, dim, source, index, inner, path, blocks);
      }
    };
    if (index.scalar_type() == at::kInt) {
      launch(int32_t{});
    } else {
      launch(int64_t{});
    }
  });
  return true;
}

} // namespace at::native::ppu_scatter_add_opt

#endif
