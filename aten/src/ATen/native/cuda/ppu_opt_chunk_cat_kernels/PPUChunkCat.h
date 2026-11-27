#pragma once

#if defined(USE_PPU)

#include <ATen/core/ATen_fwd.h>
#include <cstdint>

namespace at::native::ppu_chunk_cat_opt {

bool try_launch_chunk_cat_opt(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out);

}

#endif
