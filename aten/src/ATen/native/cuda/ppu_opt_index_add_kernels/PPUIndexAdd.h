#pragma once

#if defined(USE_PPU)

#include <ATen/core/ATen_fwd.h>
#include <cstdint>

namespace at::native::ppu_index_add_opt {

bool try_launch_index_add_opt(
    const Tensor& output,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha);

} // namespace at::native::ppu_index_add_opt

#endif
