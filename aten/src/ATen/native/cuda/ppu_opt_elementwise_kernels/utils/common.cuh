#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

// NOTE: no `namespace at::native` wrapper here. Every includer of this file
// lives in the dispatch chain, which CUDALoops.cuh includes from inside its
// own `namespace at::native` block, so the contents inherit that scope.
// Re-wrapping would create a nested `at` member (at::native::at::native)
// that shadows every `at::` qualified reference in the includer.

// Per-thread vector width in elements for same-dtype kernels (nocast
// family). 16 bytes is the widest useful access, so bf16 gives 8 (one
// 16-byte access). Note this deliberately does NOT copy
// launch_vectorized_kernel's extra "cap at 4 unless sm_90/sm_100" rule: that
// cap exists because upstream's own vec8 kernel body is compiled away
// outside sm_90/sm_100, which does not apply to the kernels below.
template <typename T>
constexpr int vector_access_width() {
  constexpr int by_size = 16 / sizeof(T);
  return by_size < 8 ? by_size : 8;
}

// Per-operand 16-byte vector width for cast kernels: each operand issues
// its own aligned_vector<T, per_op_vector_width<T>()> accesses (full 16
// bytes per access), and the caller takes max() across operands as the
// per-thread element count (lcm, powers of two).
template <typename T>
constexpr int per_op_vector_width() {
  return 16 / sizeof(T);
}

// aligned_vector<T, vt> can be lowered to several ISA loads/stores.  The
// dispatcher must gate every participating pointer and higher-dimension base
// stride by the width of one emitted ISA access, not by a different operand's
// element size or by the aggregate logical vector size.
template <typename T, int vt>
constexpr int64_t vector_isa_access_bytes() {
  constexpr int64_t logical_bytes = static_cast<int64_t>(sizeof(T)) * vt;
  return logical_bytes < 16 ? logical_bytes : 16;
}

static inline bool is_vector_access_aligned(
    const char* ptr,
    int64_t isa_access_bytes,
    std::initializer_list<int64_t> higher_dim_strides = {}) {
  if (isa_access_bytes <= 0 ||
      reinterpret_cast<uintptr_t>(ptr) % isa_access_bytes != 0) {
    return false;
  }
  for (const int64_t stride : higher_dim_strides) {
    if (stride % isa_access_bytes != 0) {
      return false;
    }
  }
  return true;
}

template <typename scalar_t, int vt>
static inline bool is_vector_access_aligned(
    const char* ptr,
    std::initializer_list<int64_t> higher_dim_strides = {}) {
  return is_vector_access_aligned(
      ptr, vector_isa_access_bytes<scalar_t, vt>(), higher_dim_strides);
}

// Vectorized load of vt contiguous T elements from an already-offset byte
// pointer (the caller guarantees sizeof(T) * vt alignment, exactly as for a
// plain aligned_vector dereference). Mirrors upstream MemoryAccess.cuh
// load_vector<vec_size, bool>: a bool tensor may hold non-canonical bytes
// (2..255, e.g. produced by view(torch.bool) / from_blob / uint8 reinterpret),
// and dereferencing such a byte as `bool` is UB that propagates the raw byte
// into casts and arithmetic (x.int() == 200). So bool is loaded as a uint8_t
// vector of the same width and each element is normalized with bool(raw),
// matching c10::load<bool> used by the legacy loops. Every other T keeps the
// original single aligned_vector dereference, i.e. identical generated code.
template <typename T, int vt>
__device__ __forceinline__ at::native::memory::aligned_vector<T, vt>
ppu_load_vector(const char* ptr) {
  if constexpr (std::is_same_v<T, bool>) {
    using raw_vec_t = at::native::memory::aligned_vector<uint8_t, vt>;
    const raw_vec_t raw = *reinterpret_cast<const raw_vec_t*>(ptr);
    at::native::memory::aligned_vector<bool, vt> ret;
#pragma unroll
    for (int i = 0; i < vt; i++) {
      ret.val[i] = static_cast<bool>(raw.val[i]);
    }
    return ret;
  } else {
    return *reinterpret_cast<const at::native::memory::aligned_vector<T, vt>*>(ptr);
  }
}
