#pragma once

// PPU elementwise aggregation header. It owns the namespace boundary when
// consumed outside at::native and combines logging with dispatch entry points.
#include "utils/log.h"

namespace at::native {
// CUDALoops.cuh defines these templates later, after its local helper
// declarations. Forward declarations preserve that definition order while
// allowing this aggregation header to remain the sole PPU include at top level.
template <int io_sizes>
constexpr auto elems_per_thread();

template <typename func_t>
constexpr auto calc_io_size();

// Every PPU elementwise kernel lives in ppu_elementwise_opt: registration.cpp
// recognizes them by this namespace and keeps them out of the runtime registry
// while disabled.
namespace ppu_elementwise_opt {
// 构建 launcher 配套启用裁剪；未使用 launcher 时保留原双架构模板与注册。
// 裁剪时 host 保留入口，只有 8.9 设备解析特化模板树，普通内核不受影响。
#if !defined(PPU_ELEMENTWISE_SM89_ONLY) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 890
#include "dispatch.cuh"
#else
template <typename func_t, typename array_t>
static inline bool try_launch_ppu_nocast_elementwise_kernel(
    TensorIteratorBase&, const func_t&, array_t) {
  return false;
}

template <typename func_t, typename array_t>
static inline bool try_launch_ppu_cast_elementwise_kernel(
    TensorIteratorBase&, const func_t&, array_t) {
  return false;
}

template <typename func_t, typename array_t>
static inline bool try_launch_ppu_cast_contiguous_unrolled_kernel(
    TensorIteratorBase&, const func_t&, array_t) {
  return false;
}
#endif
} // namespace ppu_elementwise_opt

using ppu_elementwise_opt::try_launch_ppu_cast_contiguous_unrolled_kernel;
using ppu_elementwise_opt::try_launch_ppu_cast_elementwise_kernel;
using ppu_elementwise_opt::try_launch_ppu_nocast_elementwise_kernel;
} // namespace at::native
