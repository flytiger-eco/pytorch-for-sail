// File: ppu_opt_elementwise_kernels/cast_contiguous_unrolled/ndim1_narity1_kernels/dispatch.cuh
// Dispatch: try_launch_ppu_cast_contiguous_1_1
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/log.h"
#include "contiguous_castin1.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_ppu_cast_contiguous_1_1(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;

  // launch_unrolled_kernel): 1D fully contiguous arity-1 cast copy with a
  // hardcoded {dst, src} type table covering the C1 patterns. Any mismatch
  // falls through to the original unrolled path.
  if constexpr (traits::arity == 1 &&
                (std::is_same_v<res_t, float> ||
                 std::is_same_v<res_t, c10::BFloat16> ||
                 std::is_same_v<res_t, double> ||
                 std::is_same_v<res_t, c10::complex<float>> ||
                 std::is_same_v<res_t, int64_t> ||
                 std::is_same_v<res_t, int32_t> ||
                 std::is_same_v<res_t, bool>)) {
    const int64_t numel = iter.numel();
    if (numel <= 0 || iter.ndim() != 1 || !iter.is_contiguous()) {
      return false;
    }
    const ScalarType dst_dt = iter.dtype(0);
    const ScalarType src_dt = iter.dtype(1);

    // Log only after the launcher accepts the vector-width and alignment checks.
#define PPU_CAST_CONTIGUOUS_DISP(dst_cpp, src_cpp, dst_st, src_st) \
    if (dst_dt == ScalarType::dst_st && src_dt == ScalarType::src_st) { \
      const bool launched = launch_unrolled_elementwise_kernel_1_1_contiguous_castin1<dst_cpp, src_cpp>(iter, f, data, numel); \
      if (launched) log_elementwise_info(iter, "p_e_ppu_1_1_cast_v", f); \
      return launched; \
    }
    // dst float cluster
    PPU_CAST_CONTIGUOUS_DISP(float, c10::BFloat16, Float, BFloat16)
    // K11: Float<-Half contiguous copy (wan21 C1/C2 patterns).
    PPU_CAST_CONTIGUOUS_DISP(float, c10::Half, Float, Half)
    PPU_CAST_CONTIGUOUS_DISP(float, int64_t, Float, Long)
    PPU_CAST_CONTIGUOUS_DISP(float, double, Float, Double)
    PPU_CAST_CONTIGUOUS_DISP(float, bool, Float, Bool)
    PPU_CAST_CONTIGUOUS_DISP(float, int32_t, Float, Int)
    // dst BFloat16 cluster
    PPU_CAST_CONTIGUOUS_DISP(c10::BFloat16, double, BFloat16, Double)
    PPU_CAST_CONTIGUOUS_DISP(c10::BFloat16, uint8_t, BFloat16, Byte)
    // dst double cluster
    PPU_CAST_CONTIGUOUS_DISP(double, c10::BFloat16, Double, BFloat16)
    // dst complex<float> cluster
    PPU_CAST_CONTIGUOUS_DISP(c10::complex<float>, c10::complex<double>, ComplexFloat, ComplexDouble)
    // dst int64 cluster
    PPU_CAST_CONTIGUOUS_DISP(int64_t, float, Long, Float)
    PPU_CAST_CONTIGUOUS_DISP(int64_t, bool, Long, Bool)
    // K17: Long<-Int contiguous cast copy (new_round).
    PPU_CAST_CONTIGUOUS_DISP(int64_t, int32_t, Long, Int)
    // dst int32 cluster
    PPU_CAST_CONTIGUOUS_DISP(int32_t, float, Int, Float)
    PPU_CAST_CONTIGUOUS_DISP(int32_t, int64_t, Int, Long)
    // K15: Int<-BFloat16 contiguous cast (qwen3 [65536] C-pattern).
    PPU_CAST_CONTIGUOUS_DISP(int32_t, c10::BFloat16, Int, BFloat16)
    // K18: Int<-Bool contiguous cast copy (new_round).
    PPU_CAST_CONTIGUOUS_DISP(int32_t, bool, Int, Bool)
    // dst bool cluster
    PPU_CAST_CONTIGUOUS_DISP(bool, int32_t, Bool, Int)
    PPU_CAST_CONTIGUOUS_DISP(bool, int64_t, Bool, Long)
#undef PPU_CAST_CONTIGUOUS_DISP
  }
  return false;
}
