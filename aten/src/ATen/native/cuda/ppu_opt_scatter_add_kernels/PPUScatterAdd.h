#pragma once

#if defined(USE_PPU)

#include <ATen/core/ATen_fwd.h>
#include <cstdint>

namespace at::native::ppu_scatter_add_opt {

bool try_launch_scatter_add_opt(
    const Tensor& output,
    int64_t dim,
    const Tensor& index,
    const Tensor& source);

} // namespace at::native::ppu_scatter_add_opt

#endif
