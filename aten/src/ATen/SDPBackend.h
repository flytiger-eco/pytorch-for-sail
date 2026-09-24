#pragma once
#include <cstdint>

namespace at {

// PPU modification: flex flash attention SDPA backend adds one
// slot (guarded by USE_PPU, see Note [PPU SDPA backends]).
#ifdef USE_PPU
constexpr int32_t num_sdp_backends = 6;
#else
constexpr int32_t num_sdp_backends = 5;
#endif
enum class SDPBackend {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2,
  cudnn_attention = 3,
  overrideable = 4,
#ifdef USE_PPU
  // PPU modification: FA3 flex flash attention backend (libflex_flash_attention.so).
  flex_flash_attention = 5
#endif
};

} // namespace at
