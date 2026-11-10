#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/ppu_opt_index_add_kernels/PPUIndexAdd.h>

#if defined(USE_PPU)

#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/NonEmptyUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/util/env.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace at::native::ppu_index_add_opt {

inline constexpr int kIndexAddThreads = 256;
inline constexpr int kIndexAddBlocksPerSM = 8;

enum class IndexAddPath {
  Native,
  Contiguous,
};

// Keep performance admission separate from the public operator's supported layouts.
// New structural fast paths belong here; unmatched structures must use native routing.
static IndexAddPath select_index_add_path(
    bool contiguous,
    int64_t dim,
    int64_t inner,
    bool aligned) {
  if (contiguous && dim == 0 && inner % 2 == 0 && aligned) {
    return IndexAddPath::Contiguous;
  }
  return IndexAddPath::Native;
}

template <typename scalar_t>
struct IndexAddAtomicTraits;

template <>
struct IndexAddAtomicTraits<at::Half> {
  using native_type = __half;
  using pair_type = __half2;
};

template <>
struct IndexAddAtomicTraits<at::BFloat16> {
  using native_type = __nv_bfloat16;
  using pair_type = __nv_bfloat162;
};

static_assert(sizeof(__half2) == sizeof(uint32_t));
static_assert(sizeof(__nv_bfloat162) == sizeof(uint32_t));

// Read two valid source values per thread; each source row shares one destination row.
template <typename scalar_t, typename offset_t, typename index_t, bool Scale>
__global__ void index_add_pair_kernel(
    scalar_t* output,
    const typename IndexAddAtomicTraits<scalar_t>::pair_type* source,
    const index_t* index,
    offset_t pairs,
    offset_t half_width,
    offset_t output_rows,
    offset_t output_numel,
    offset_t index_row_stride,
    scalar_t alpha) {
  using native_t = typename IndexAddAtomicTraits<scalar_t>::native_type;
  using pair_t = typename IndexAddAtomicTraits<scalar_t>::pair_type;
  const offset_t grid_stride = static_cast<offset_t>(blockDim.x) * gridDim.x;
  for (offset_t q = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       q < pairs; q += grid_stride) {
    const offset_t row = q / half_width;
    const offset_t column = (q % half_width) * 2;
    const offset_t index_offset = row * index_row_stride;
    const int64_t dst0 = index[index_offset];
    CUDA_KERNEL_ASSERT(dst0 >= 0 && dst0 < static_cast<int64_t>(output_rows));
    auto value = source[q];
    if constexpr (Scale) {
      // Match the original index_add multiplication: multiply in FP32, then round each product to the tensor dtype.
      value.x = static_cast<native_t>(scalar_t(value.x) * alpha);
      value.y = static_cast<native_t>(scalar_t(value.y) * alpha);
    }
    const offset_t offset0 =
        static_cast<offset_t>(dst0) * (half_width * 2) + column;
    const offset_t offset1 = offset0 + 1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
    // Use the SDK pair overload without adding assembly or changing SDK gates.
    atomicAdd(reinterpret_cast<pair_t*>(output + offset0), value);
    continue;
#endif
    fastAtomicAdd(output, offset0, output_numel, scalar_t(value.x), true);
    fastAtomicAdd(output, offset1, output_numel, scalar_t(value.y), true);
  }
}

template <typename scalar_t, typename offset_t, typename index_t, bool Scale = false>
static void launch_index_add_pair(
    const Tensor& output,
    const Tensor& source,
    const Tensor& index,
    offset_t pairs,
    offset_t half_width,
    int64_t index_row_stride,
    int blocks,
    scalar_t alpha = scalar_t(1.0f)) {
  using pair_t = typename IndexAddAtomicTraits<scalar_t>::pair_type;
  const auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  index_add_pair_kernel<scalar_t, offset_t, index_t, Scale>
      <<<blocks, kIndexAddThreads, 0, stream>>>(
      output.mutable_data_ptr<scalar_t>(),
      reinterpret_cast<const pair_t*>(source.const_data_ptr<scalar_t>()),
      index.const_data_ptr<index_t>(),
      pairs,
      half_width,
      static_cast<offset_t>(output.size(0)),
      static_cast<offset_t>(output.numel()),
      static_cast<offset_t>(index_row_stride),
      alpha);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename offset_t, typename index_t>
static void dispatch_index_add(
    const Tensor& output,
    const Tensor& source,
    const Tensor& index,
    int64_t inner,
    int blocks,
    const Scalar& alpha) {
  const auto pairs = static_cast<offset_t>(source.numel() / 2);
  const auto half_width = static_cast<offset_t>(inner / 2);
  if (!alpha.isSymbolic() && alpha.equal(1)) {
    launch_index_add_pair<scalar_t, offset_t, index_t, false>(
        output, source, index, pairs, half_width, ensure_nonempty_stride(index, 0), blocks);
  } else {
    launch_index_add_pair<scalar_t, offset_t, index_t, true>(
        output, source, index, pairs, half_width, ensure_nonempty_stride(index, 0), blocks,
        alpha.to<scalar_t>());
  }
}

// Called only after native CUDA index_add meta validation and deterministic routing.
// The dispatcher handles negative bits, wrappers check devices, and meta validates dtype and shape consistency.
bool try_launch_index_add_opt(
    const Tensor& output,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha) {
  static const bool enabled =
      !c10::utils::check_env("PYTORCH_DISABLE_PPU_INDEX_ADD_OPT").value_or(false);
  if (!enabled) {
    return false;
  }
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
  if (alpha.isComplex()) {
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
  const bool aligned =
      reinterpret_cast<std::uintptr_t>(output.const_data_ptr()) % sizeof(uint32_t) == 0 &&
      reinterpret_cast<std::uintptr_t>(source.const_data_ptr()) % sizeof(uint32_t) == 0;
  const auto path = select_index_add_path(contiguous, dim, inner, aligned);
  if (path == IndexAddPath::Native) {
    return false;
  }
  const int64_t numel = source.numel();
  const int64_t pairs = numel / 2 + numel % 2;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (pairs + kIndexAddThreads - 1) / kIndexAddThreads,
      std::min<int64_t>(
          static_cast<int64_t>(prop->multiProcessorCount) * kIndexAddBlocksPerSM,
          prop->maxGridSize[0])));
  // Index values retain their storage dtype; all three tensors use the selected offset width.
  const bool use_32bit =
      canUse32BitIndexMath(output) && canUse32BitIndexMath(source) &&
      canUse32BitIndexMath(index);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(output.scalar_type(), "ppu_index_add_pair", [&] {
    const auto launch = [&](auto index_type) {
      using index_t = decltype(index_type);
      if (use_32bit) {
        dispatch_index_add<scalar_t, uint32_t, index_t>(
            output, source, index, inner, blocks, alpha);
      } else {
        dispatch_index_add<scalar_t, uint64_t, index_t>(
            output, source, index, inner, blocks, alpha);
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

} // namespace at::native::ppu_index_add_opt

#endif
