#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/ppu_opt_index_scatter_kernels/PPUIndexScatter.h>

#if defined(USE_PPU)

#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/NonEmptyUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace at::native::ppu_index_scatter_opt {

inline constexpr int kPairAtomicThreads = 256;
inline constexpr int kPairAtomicBlocksPerSM = 8;

enum class PairAtomicPath {
  Native,
  ContiguousIndex,
  BroadcastScatter,
  DenseScatter,
  IndexRankCompatibility,
};

// Keep performance admission separate from the public operator's supported layouts.
// New structural fast paths belong here; unmatched structures must use native routing.
template <bool Scatter>
static PairAtomicPath select_pair_atomic_path(
    bool contiguous,
    int64_t dim,
    int64_t inner,
    bool aligned,
    bool broadcast,
    bool full_dense,
    int64_t rank) {
  if (contiguous && dim == 0 && inner % 2 == 0 && aligned) {
    if constexpr (!Scatter) {
      return PairAtomicPath::ContiguousIndex;
    } else {
      if (broadcast) {
        return PairAtomicPath::BroadcastScatter;
      }
      if (full_dense) {
        // Dense indices may target different rows; the kernel checks each pair before combining.
        return PairAtomicPath::DenseScatter;
      }
    }
  }
  if constexpr (!Scatter) {
    // Native TensorInfo rejects rank == MAX_TENSORINFO_DIMS before collapsing singleton axes.
    // Retain the existing ND implementation only as a compatibility route at that boundary.
    if (rank == MAX_TENSORINFO_DIMS) {
      return PairAtomicPath::IndexRankCompatibility;
    }
  }
  return PairAtomicPath::Native;
}

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
template <typename scalar_t, typename offset_t, typename index_t, bool Scatter, bool Scale>
__global__ void pair_atomic_add_kernel(
    scalar_t* output,
    const typename PairAtomicTraits<scalar_t>::pair_type* source,
    const index_t* index,
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

template <typename scalar_t, typename offset_t, typename index_t, bool Scatter, bool Scale = false>
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
  pair_atomic_add_kernel<scalar_t, offset_t, index_t, Scatter, Scale>
      <<<blocks, kPairAtomicThreads, 0, stream>>>(
      output.mutable_data_ptr<scalar_t>(),
      reinterpret_cast<const pair_t*>(source.const_data_ptr<scalar_t>()),
      index.const_data_ptr<index_t>(),
      pairs,
      half_width,
      static_cast<offset_t>(output.size(0)),
      static_cast<offset_t>(output.numel()),
      static_cast<offset_t>(index_row_stride),
      static_cast<offset_t>(index_column_stride),
      alpha);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Continuous index_add is represented as outer x indexed x inner, independently of tensor rank.
template <typename offset_t, bool DenseScatter = false>
struct IndexAddOffsetCalculator {
  offset_t inner;
  offset_t source_dim_size;
  offset_t output_dim_size;
  offset_t index_stride;

  C10_HOST_DEVICE std::array<offset_t, 3> get(offset_t linear) const {
    const offset_t slice = linear / inner;
    const offset_t indexed = slice % source_dim_size;
    const offset_t outer = slice / source_dim_size;
    return {outer * output_dim_size * inner + linear % inner,
            linear, DenseScatter ? linear : indexed * index_stride};
  }
};

// A pair is only issued for two valid, adjacent, aligned destinations; tails use a single update.
template <typename scalar_t, typename offset_t, typename index_t, typename calculator_t, bool Scale>
__global__ void pair_atomic_add_nd_kernel(
    scalar_t* output,
    const scalar_t* source,
    const index_t* index,
    offset_t numel,
    offset_t output_numel,
    offset_t output_dim_size,
    offset_t output_dim_stride,
    calculator_t calculator,
    scalar_t alpha,
    bool fast_atomics) {
  using native_t = typename PairAtomicTraits<scalar_t>::native_type;
  using pair_t = typename PairAtomicTraits<scalar_t>::pair_type;
  const offset_t pairs = numel / 2 + numel % 2;
  const offset_t grid_stride = static_cast<offset_t>(blockDim.x) * gridDim.x;
  for (offset_t q = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       q < pairs; q += grid_stride) {
    const offset_t linear = q * 2;
    const auto offsets0 = calculator.get(linear);
    const int64_t dst0 = index[offsets0[2]];
    CUDA_KERNEL_ASSERT(dst0 >= 0 && dst0 < static_cast<int64_t>(output_dim_size));
    const offset_t offset0 = offsets0[0] + static_cast<offset_t>(dst0) * output_dim_stride;
    if (linear + 1 == numel) {
      scalar_t value = source[offsets0[1]];
      if constexpr (Scale) {
        value = value * alpha;
      }
      fastAtomicAdd(output, offset0, output_numel, value, fast_atomics);
      continue;
    }
    const auto offsets1 = calculator.get(linear + 1);
    const int64_t dst1 = index[offsets1[2]];
    CUDA_KERNEL_ASSERT(dst1 >= 0 && dst1 < static_cast<int64_t>(output_dim_size));
    const offset_t offset1 = offsets1[0] + static_cast<offset_t>(dst1) * output_dim_stride;
    pair_t value;
    if (offsets1[1] == offsets0[1] + 1 &&
        reinterpret_cast<std::uintptr_t>(source + offsets0[1]) % sizeof(uint32_t) == 0) {
      value = *reinterpret_cast<const pair_t*>(source + offsets0[1]);
    } else {
      value.x = static_cast<native_t>(source[offsets0[1]]);
      value.y = static_cast<native_t>(source[offsets1[1]]);
    }
    if constexpr (Scale) {
      value.x = static_cast<native_t>(scalar_t(value.x) * alpha);
      value.y = static_cast<native_t>(scalar_t(value.y) * alpha);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
    if (offset1 == offset0 + 1 &&
        reinterpret_cast<std::uintptr_t>(output + offset0) % sizeof(uint32_t) == 0) {
      atomicAdd(reinterpret_cast<pair_t*>(output + offset0), value);
      continue;
    }
#endif
    // Do not pad a strided destination's single update with an unrelated storage neighbor.
    fastAtomicAdd(output, offset0, output_numel, scalar_t(value.x), fast_atomics);
    fastAtomicAdd(output, offset1, output_numel, scalar_t(value.y), fast_atomics);
  }
}

template <typename scalar_t, typename offset_t, typename index_t, typename calculator_t, bool Scale = false>
static void launch_pair_atomic_add_nd(
    const Tensor& output,
    int64_t dim,
    const Tensor& source,
    const Tensor& index,
    int64_t numel,
    calculator_t calculator,
    int blocks,
    scalar_t alpha = scalar_t(1.0f)) {
  const auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  pair_atomic_add_nd_kernel<scalar_t, offset_t, index_t, calculator_t, Scale>
      <<<blocks, kPairAtomicThreads, 0, stream>>>(
      output.mutable_data_ptr<scalar_t>(), source.const_data_ptr<scalar_t>(),
      index.const_data_ptr<index_t>(), static_cast<offset_t>(numel),
      static_cast<offset_t>(output.numel()),
      static_cast<offset_t>(ensure_nonempty_size(output, dim)),
      static_cast<offset_t>(ensure_nonempty_stride(output, dim)), calculator, alpha,
      output.is_contiguous());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename offset_t, typename index_t, bool Scatter>
static void dispatch_pair_atomic_add(
    const Tensor& output,
    int64_t dim,
    const Tensor& source,
    const Tensor& index,
    int64_t inner,
    PairAtomicPath path,
    bool broadcast,
    bool full_dense,
    bool use_iterator,
    const TensorIterator* iterator,
    int blocks,
    const Scalar& alpha) {
  if (path == PairAtomicPath::ContiguousIndex ||
      path == PairAtomicPath::BroadcastScatter || path == PairAtomicPath::DenseScatter) {
    const auto pairs = static_cast<offset_t>(source.numel() / 2);
    const auto half_width = static_cast<offset_t>(inner / 2);
    if constexpr (Scatter) {
      if (path == PairAtomicPath::BroadcastScatter) {
        launch_pair_atomic_add<scalar_t, offset_t, index_t, false>(
            output, source, index, pairs, half_width, ensure_nonempty_stride(index, dim), 0, blocks);
      } else {
        launch_pair_atomic_add<scalar_t, offset_t, index_t, true>(
            output, source, index, pairs, half_width,
            full_dense ? inner : index.stride(0), full_dense ? 1 : index.stride(1), blocks);
      }
    } else {
      if (!alpha.isSymbolic() && alpha.equal(1)) {
        launch_pair_atomic_add<scalar_t, offset_t, index_t, false, false>(
            output, source, index, pairs, half_width, ensure_nonempty_stride(index, 0), 0, blocks);
      } else {
        launch_pair_atomic_add<scalar_t, offset_t, index_t, false, true>(
            output, source, index, pairs, half_width, ensure_nonempty_stride(index, 0), 0, blocks,
            alpha.to<scalar_t>());
      }
    }
    return;
  }
  if (use_iterator) {
    std::array<const int64_t*, 3> strides;
    std::array<int64_t, 3> element_sizes;
    for (int arg = 0; arg < 3; ++arg) {
      strides[arg] = iterator->strides(arg).data();
      element_sizes[arg] = iterator->element_size(arg);
    }
    const OffsetCalculator<3, offset_t> calculator(
        iterator->ndim(), iterator->shape().data(), strides.data(), element_sizes.data());
    if constexpr (Scatter) {
      launch_pair_atomic_add_nd<scalar_t, offset_t, index_t>(
          output, dim, source, index, iterator->numel(), calculator, blocks);
    } else {
      if (!alpha.isSymbolic() && alpha.equal(1)) {
        launch_pair_atomic_add_nd<scalar_t, offset_t, index_t, decltype(calculator), false>(
            output, dim, source, index, iterator->numel(), calculator, blocks);
      } else {
        launch_pair_atomic_add_nd<scalar_t, offset_t, index_t, decltype(calculator), true>(
            output, dim, source, index, iterator->numel(), calculator, blocks, alpha.to<scalar_t>());
      }
    }
    return;
  }
  if constexpr (Scatter) {
    if (full_dense) {
      const IndexAddOffsetCalculator<offset_t, true> calculator{
          static_cast<offset_t>(inner),
          static_cast<offset_t>(ensure_nonempty_size(source, dim)),
          static_cast<offset_t>(ensure_nonempty_size(output, dim)), 1};
      launch_pair_atomic_add_nd<scalar_t, offset_t, index_t>(
          output, dim, source, index, source.numel(), calculator, blocks);
      return;
    }
  }
  const IndexAddOffsetCalculator<offset_t> calculator{
      static_cast<offset_t>(inner),
      static_cast<offset_t>(ensure_nonempty_size(source, dim)),
      static_cast<offset_t>(ensure_nonempty_size(output, dim)),
      static_cast<offset_t>(ensure_nonempty_stride(index, Scatter ? dim : 0))};
  if constexpr (Scatter) {
    launch_pair_atomic_add_nd<scalar_t, offset_t, index_t>(
        output, dim, source, index, source.numel(), calculator, blocks);
  } else {
    if (!alpha.isSymbolic() && alpha.equal(1)) {
      launch_pair_atomic_add_nd<scalar_t, offset_t, index_t, decltype(calculator), false>(
          output, dim, source, index, source.numel(), calculator, blocks);
    } else {
      launch_pair_atomic_add_nd<scalar_t, offset_t, index_t, decltype(calculator), true>(
          output, dim, source, index, source.numel(), calculator, blocks, alpha.to<scalar_t>());
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
  if constexpr (Scatter) {
    if (!contiguous) {
      return false;
    }
  }
  if constexpr (!Scatter) {
    if (alpha.isComplex()) {
      return false;
    }
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
  bool broadcast = false;
  bool full_shape = true;
  if constexpr (Scatter) {
    full_shape = index.sizes() == source.sizes();
    for (int64_t d = 0; d < output.dim(); ++d) {
      if (d != dim && output.size(d) != source.size(d)) {
        full_shape = false;
      }
    }
    broadcast = full_shape;
    for (int64_t d = 0; d < index.dim(); ++d) {
      if (d != dim && index.size(d) > 1 && index.stride(d) != 0) {
        broadcast = false;
      }
    }
  }
  const bool aligned =
      reinterpret_cast<std::uintptr_t>(output.const_data_ptr()) % sizeof(uint32_t) == 0 &&
      reinterpret_cast<std::uintptr_t>(source.const_data_ptr()) % sizeof(uint32_t) == 0;
  const bool full_dense = Scatter && full_shape && index.is_contiguous();
  const auto path = select_pair_atomic_path<Scatter>(
      contiguous, dim, inner, aligned, broadcast, full_dense, output.dim());
  if (path == PairAtomicPath::Native) {
    return false;
  }
  // Iterator setup is reserved for the explicit rank-boundary compatibility route.
  const bool use_iterator = path == PairAtomicPath::IndexRankCompatibility && !contiguous;
  TensorIterator iterator;
  Tensor output_restrided;
  Tensor source_restrided;
  Tensor index_restrided;
  if (use_iterator) {
    // Preserve true source/output strides; the destination's indexed axis is supplied by index values.
    auto sizes = ensure_nonempty_vec((Scatter ? index : source).sizes().vec());
    auto output_strides = ensure_nonempty_vec(output.strides().vec());
    output_strides[dim] = 0;
    output_restrided = output.as_strided(sizes, output_strides);
    source_restrided = source.as_strided(sizes, ensure_nonempty_vec(source.strides().vec()));
    if constexpr (Scatter) {
      index_restrided = index;
    } else {
      std::vector<int64_t> index_strides(sizes.size(), 0);
      index_strides[dim] = ensure_nonempty_stride(index, 0);
      index_restrided = index.as_strided(sizes, index_strides);
    }
    iterator = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .add_output(output_restrided)
        .add_const_input(source_restrided)
        .add_const_input(index_restrided)
        .build();
    // Use the native offset calculator's coalesced-dimension limit, not an original-rank restriction.
    if (iterator.ndim() > MAX_DIMS) {
      return false;
    }
  }
  const int64_t numel = use_iterator ? iterator.numel() : source.numel();
  const int64_t pairs = numel / 2 + numel % 2;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (pairs + kPairAtomicThreads - 1) / kPairAtomicThreads,
      std::min<int64_t>(
          static_cast<int64_t>(prop->multiProcessorCount) * kPairAtomicBlocksPerSM,
          prop->maxGridSize[0])));
  // Index values retain their storage dtype; all three tensors use the selected offset width.
  const bool use_32bit =
      canUse32BitIndexMath(output) && canUse32BitIndexMath(source) &&
      canUse32BitIndexMath(index);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(output.scalar_type(), "ppu_pair_atomic_add", [&] {
    const auto launch = [&](auto index_type) {
      using index_t = decltype(index_type);
      if (use_32bit) {
        dispatch_pair_atomic_add<scalar_t, uint32_t, index_t, Scatter>(
            output, dim, source, index, inner, path, broadcast, full_dense,
            use_iterator, &iterator, blocks, alpha);
      } else {
        dispatch_pair_atomic_add<scalar_t, uint64_t, index_t, Scatter>(
            output, dim, source, index, inner, path, broadcast, full_dense,
            use_iterator, &iterator, blocks, alpha);
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
