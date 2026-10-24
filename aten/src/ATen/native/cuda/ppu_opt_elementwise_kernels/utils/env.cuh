#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>

#include <cuda_runtime_api.h>

namespace at::native {

struct PPUDeviceCapabilities {
  bool available{false};
  bool is_m890{false};
  std::array<int64_t, 3> max_grid_size{0, 0, 0};
};

// Query the device selected on first use once per process. Elementwise dispatch
// assumes the usual one-process-per-device execution model and does not support
// switching to a different device model after this cache has been initialized.
inline const PPUDeviceCapabilities& current_ppu_device_capabilities() {
  static const PPUDeviceCapabilities capabilities = []() {
    PPUDeviceCapabilities result;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
      return result;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
      return result;
    }

    result.available = true;
    result.is_m890 = (prop.major == 8 && prop.minor == 9);
    for (int axis = 0; axis < 3; ++axis) {
      result.max_grid_size[axis] = prop.maxGridSize[axis];
    }
    return result;
  }();
  return capabilities;
}

inline bool is_m890_device() {
  const auto& capabilities = current_ppu_device_capabilities();
  return capabilities.available && capabilities.is_m890;
}

// Returns zero when the current device cannot be queried; callers treat that
// as an ineligible launch rather than constructing an unchecked dim3 value.
inline int64_t ppu_max_grid_size(int axis) {
  const auto& capabilities = current_ppu_device_capabilities();
  if (!capabilities.available || axis < 0 || axis >= 3) {
    return 0;
  }
  return capabilities.max_grid_size[axis];
}

// Empirical sweet cap on host-side grid block counts (2^16 - 1). 65535 is
// both the historical gridDim limit and the measured sweet spot: below it
// per-block scheduling stays efficient, above it the per-block overhead
// dominates and the wider-grid elementwise variants regress by 1.3-3.6x
// (measured on the 2_2/3_1/3_2 families at 4M-64M). Tune this single macro
// instead of the per-site min() expressions.
#define PPU_GRID_SWEET_CAP 65535

// Convenience clamp for older call sites: the device's gridDim limit
// tightened by the empirical sweet cap above. New code spells the two
// bounds out at each gate instead - std::min<int64_t>(ppu_max_grid_size(
// axis), PPU_GRID_SWEET_CAP) - so the hardware limit and the performance
// sweet spot stay distinguishable per site.
inline int64_t ppu_grid_cap(int axis) {
  return std::min<int64_t>(ppu_max_grid_size(axis), PPU_GRID_SWEET_CAP);
}

// Whether PPU elementwise kernels were registered when torch_cuda was loaded.
// registration.cpp enables them only when every visible device is a Zhenwu M890
// and PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT="1" was set before loading torch_cuda.
bool elementwise_ppu_kernels_registered();

// Follow the load-time registration decision rather than re-reading the
// environment or querying the device from the hot path. A dropped kernel must
// never be launched.
inline bool elementwise_ppu_enabled() {
  return elementwise_ppu_kernels_registered();
}

static inline const std::string& local_rank() {
  static const std::string rank = []() {
    if (const char* value = std::getenv("LOCAL_RANK")) {
      return std::string(value);
    }
    if (const char* value = std::getenv("RANK")) {
      return std::string(value);
    }
    return std::string("0");
  }();
  return rank;
}

} // namespace at::native
