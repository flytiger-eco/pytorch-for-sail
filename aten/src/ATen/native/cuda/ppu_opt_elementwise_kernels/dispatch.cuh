// File: ppu_opt_elementwise_kernels/dispatch.cuh
// PPU elementwise dispatch entry points.
// Per-ndim/arity predicates live in the corresponding subdirectories.

#pragma once

#include "utils/env.cuh"
#include "nocast_elementwise/include.cuh"
#include "cast_elementwise/include.cuh"
#include "cast_contiguous_unrolled/include.cuh"

// ── Dispatch table macros ────────────────────────────────────
// X-macro tables eliminate the manual if/else chain in each entry
// function.  The helper expands to a single arity+ndim check.
#define PPU_DISPATCH_ONE(PREFIX, NDIM, ARITY)                    \
  if constexpr (traits::arity == ARITY) {                        \
    if (iter.ndim() == NDIM) {                                   \
      return PREFIX##_##NDIM##_##ARITY(iter, f, data);           \
    }                                                            \
  }

// nocast_elementwise table (arity 1..3, ndim 1..5).
#define PPU_NOCAST_DISPATCH_TABLE(F)                             \
  F(try_launch_ppu, 1, 1) F(try_launch_ppu, 2, 1)                \
  F(try_launch_ppu, 3, 1) F(try_launch_ppu, 4, 1)                \
  F(try_launch_ppu, 5, 1) F(try_launch_ppu, 1, 2)                \
  F(try_launch_ppu, 2, 2) F(try_launch_ppu, 3, 2)                \
  F(try_launch_ppu, 4, 2) F(try_launch_ppu, 1, 3)

// cast_elementwise table (arity 1..2, ndim 1..3).
#define PPU_CAST_DISPATCH_TABLE(F)                               \
  F(try_launch_ppu_cast_elementwise, 2, 1)                        \
  F(try_launch_ppu_cast_elementwise, 1, 2)                        \
  F(try_launch_ppu_cast_elementwise, 2, 2)                        \
  F(try_launch_ppu_cast_elementwise, 3, 2)

// cast_contiguous_unrolled table (arity 1..2, ndim 1).
#define PPU_CAST_CONTIG_DISPATCH_TABLE(F)                        \
  F(try_launch_ppu_cast_contiguous, 1, 1)                         \
  F(try_launch_ppu_cast_contiguous, 1, 2)

// Shared contract:
//   - vector operands are aligned and contiguous on dim0;
//   - vector accesses do not cross row boundaries;
//   - stride-0 operands use scalar loads;
//   - mismatches fall through to the legacy path.

template <typename func_t, typename array_t>
static inline bool try_launch_ppu_nocast_elementwise_kernel(TensorIteratorBase& iter, const func_t& f, array_t data) {
  if (!elementwise_ppu_enabled()) return false;
  using traits = function_traits<func_t>;
  PPU_NOCAST_DISPATCH_TABLE(PPU_DISPATCH_ONE)
  return false;
}

template <typename func_t, typename array_t>
static inline bool try_launch_ppu_cast_elementwise_kernel(TensorIteratorBase& iter, const func_t& f, array_t data) {
  if (!elementwise_ppu_enabled()) return false;
  using traits = function_traits<func_t>;
  PPU_CAST_DISPATCH_TABLE(PPU_DISPATCH_ONE)
  return false;
}

template <typename func_t, typename array_t>
static inline bool try_launch_ppu_cast_contiguous_unrolled_kernel(TensorIteratorBase& iter, const func_t& f, array_t data) {
  if (!elementwise_ppu_enabled()) return false;
  using traits = function_traits<func_t>;
  PPU_CAST_CONTIG_DISPATCH_TABLE(PPU_DISPATCH_ONE)
  return false;
}
