/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <mutex>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <iostream>
#include <algorithm>


#include <ATen/native/transformers/cuda/mem_eff_attention/mem_eff_api.h>

// cutlass related include
// moved from attention_forward.cu
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassF.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/pytorch_utils.h>
// moved from attention_backward.cu
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassB.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h>

// #define DEBUG

namespace at {

namespace native {
// namespace {

/*
  There are 2 modes for using this function.
  (Mode BMHK) With all the heads having the same seqlen
  (Mode 1MHK) `batch=1` with all tokens across batches concatenated
*/
std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt>
efficient_attention_forward_cutlass_origin(
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    const std::optional<at::Tensor>& bias, // [b, num_heads, seqlen, seqlen]
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const std::optional<at::Tensor>& seqstart_q,
    // (Mode 1MHK only) [b+1]: cu_seqlen_k[b] contains the
    // position of the first key token for batch $b
    const std::optional<at::Tensor>& seqstart_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const std::optional<int64_t> max_seqlen_q_,
    const std::optional<int64_t> max_seqlen_k_,
    double dropout_p, // attention matrix dropout probability
    int64_t custom_mask_type,
    bool compute_logsumexp,
    std::optional<double> scale,
    const std::optional<at::Tensor>& seqlen_k,
    const std::optional<int64_t> window_size) {
// TODO In theory it is possible to compile with _CUDA_ARCH < 5.0 and run on a
// machine that is >= 5.0. In practice, this is not a problem but since
// this would avoid runtime architecture checks, we should look into it

  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(key.dim() == 4);
  TORCH_CHECK(value.dim() == 4);

  // Batch sizes
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // Sequence length
  TORCH_CHECK(key.size(1) == value.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));

  int64_t max_seqlen_q = 0, max_seqlen_k = 0;
  TORCH_CHECK(seqstart_q.has_value() == seqstart_k.has_value());
  if (seqstart_q.has_value()) {
    TORCH_CHECK(seqstart_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_q->dim() == 1 && seqstart_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_k));
    TORCH_CHECK(seqstart_q->size(0) == seqstart_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q_.has_value());
    max_seqlen_q = *max_seqlen_q_;
    max_seqlen_k = 0; // TODO: is this actually being set inside the kernel anywhere?
                      // see https://github.com/pytorch/pytorch/issues/115590s
  } else {
    max_seqlen_q = query.size(1);
    max_seqlen_k = key.size(1);
  }

  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t num_heads = query.size(-2);
  int64_t K = query.size(-1);
  int64_t Kv = value.size(-1);

  at::Tensor res;
  at::Tensor logsumexp;
  at::Tensor seed_t, offset_t;

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;

  // Note [Seed and Offset Device]
  // If we are currently in graph capture mode, we need to create the seed and offset tensors on the device.
  // This is necessary for CUDA graph-safe random number generation, which requires the seed and offset tensors
  // to be single element tensors on device. During graph capture, when the seed and offset tensors are passed
  // the pointers act as scratch space for storing the RNG state for the backwards pass.
  // When calling backwards, we either construct a PhiloxState with the pointers or the actual values.
  // For more information on CUDA graph-safe RNG states, see Note [CUDA Graph-safe RNG states].

  at::PhiloxCudaState philox_state;
  const bool in_capture_stream =
      at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None;
  auto device = in_capture_stream ? at::kCUDA : at::kCPU;
  if (use_dropout) {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    // if using dropout, we produce 1 random number for each element of the
    // attention tensor
    philox_state = gen->philox_cuda_state(B * num_heads * M * N);

    if (in_capture_stream) {
      // The seed and offset will be populated by the kernel
      seed_t = at::empty({}, at::dtype(at::kLong).device(device));
      offset_t = at::empty({}, at::dtype(at::kLong).device(device));
    } else {
      auto [seed, offset] = at::cuda::philox::unpack(philox_state);
#ifdef USE_ROCM
      const auto options = at::dtype(at::kLong).device(at::kCUDA);
#else
      const auto options = at::dtype(at::kLong);
#endif
      seed_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(seed)), options);
      offset_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(offset)), options);
    }
  } else {
    // Not using dropout
    seed_t = at::empty({}, at::dtype(at::kLong).device(device));
    offset_t = at::empty({}, at::dtype(at::kLong).device(device));
  }

#ifdef USE_ROCM
  // ROCM Implementation

  // Need this in both aot and CK case
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  res = at::empty({B, M, num_heads, Kv}, query.options());

  if(at::globalContext().getROCmFAPreferredBackend() ==
    at::ROCmFABackend::Ck) {

#if defined(USE_ROCM_CK_SDPA)
    std::optional<Tensor> out(res);
    std::optional<Tensor> seqused_k = std::nullopt;
    std::optional<Tensor> alibi_slopes = std::nullopt;
    auto
        [out_,
         q,
         k,
         v,
         lse,
         seed_t,
         offset_t,
         p] =
            pytorch_flash::mem_eff_forward_ck(
                                    query,
                                    key,
                                    value,
                                    dropout_p,
                                    false,                                // return dropout_randval
                                    custom_mask_type == 0 ? false : true, // is_causal
                                    softmax_scale,
                                    bias,
                                    out,
                                    std::nullopt,                         // cu_seqlens_q
                                    std::nullopt,                         // cu_seqlens_k
                                    seqstart_q,
                                    seqstart_k,
                                    std::nullopt,                         // gen_
                                    seqused_k);                           // seqused_k_

    logsumexp = lse;
#else
    TORCH_CHECK(false, "Attempting to use CK mem_eff_forward backend in a build that has not built CK");
#endif
  } else { // use aotriton
#ifndef DISABLE_AOTRITON
    auto ret = aotriton::v2::flash::check_gpu(stream);
    if (hipSuccess != ret) {
        TORCH_CHECK(false,
                  "[AOTriton] Accelerated SDPA only supports MI200/MI300X/Navi31 GPUs"
                  " (gfx90a:sramecc+:xnack-/gfx942:sramecc+:xnack-/gfx1100)")
    }

    // AOTriton may accept aligned on logsumexp tensor in the future for better
    // performance, but for now it requires compact logsumexp tensor, even if
    // compute_logsumexp is false
    constexpr int kAlignLSE = 1;
    res = at::empty({B, M, num_heads, Kv}, query.options());
    at::Tensor softmax_lse;
    logsumexp = at::empty(
      { B, num_heads, compute_logsumexp ? max_seqlen_q : 0},
      query.options().dtype(at::ScalarType::Float));
    if (compute_logsumexp) {
      softmax_lse = logsumexp.view({B * num_heads, max_seqlen_q});
    }
    at::Tensor q_t = query.transpose(1, 2);
    at::Tensor k_t = key.transpose(1, 2);
    at::Tensor v_t = value.transpose(1, 2);
    at::Tensor output_t = res.transpose(1, 2);
    bool is_causal;
    if (static_cast<int64_t>(sdp::CustomMaskType::NoCustomMask) == custom_mask_type) {
      is_causal = false;
    } else {
      is_causal = true;
#if AOTRITON_V3_API == 0
      if (static_cast<int64_t>(sdp::CustomMaskType::CausalFromTopLeft) != custom_mask_type) {
        TORCH_CHECK(false, "[_efficient_attention_forward] Unsupported mask type on ROCM, for now");
      }
#endif
    }

    at::Tensor atomic_counter;
    if (is_causal) {
      atomic_counter = at::zeros({1}, query.options().dtype(at::kInt));
    }

    using aotriton::v2::flash::attn_fwd;
    using aotriton::v2::flash::attn_fwd_compact_varlen;
    using sdp::aotriton_adapter::mk_aotensor;
    using sdp::aotriton_adapter::mk_aoscalartensor;
    using sdp::aotriton_adapter::mk_philoxtensor;
    using sdp::aotriton_adapter::mk_atomictensor;
    aotriton::TensorView<4> empty_t4(0, {0, 0, 0, 0}, {0, 0, 0, 0}, aotriton::DType::kFloat16);
    aotriton::TensorView<2> empty_t2(0, {0, 0}, {0, 0}, aotriton::DType::kFloat32);
    at::Tensor softmax_fa_t = at::empty({ 0, 0, 0, 0 }, query.options());
    const bool use_philox_state = in_capture_stream;
    auto seed = use_philox_state ? mk_philoxtensor(philox_state.seed_.ptr) : mk_aoscalartensor(seed_t);
    auto offset1 = use_philox_state ? mk_philoxtensor(philox_state.offset_.ptr) : mk_aoscalartensor(offset_t);
    auto offset2 = use_philox_state ? philox_state.offset_intragraph_ : 0;
    auto seed_output = mk_philoxtensor(use_philox_state ? seed_t.data_ptr<int64_t>() : nullptr);
    auto offset_output = mk_philoxtensor(use_philox_state ? offset_t.data_ptr<int64_t>() : nullptr);
    auto persistent_counter = mk_atomictensor(is_causal ? atomic_counter.data_ptr<int32_t>() : nullptr);
    hipError_t err; // TODO: Error handling
    if constexpr (AOTRITON_ALWAYS_V3_API) {  // Better readability than nesting ifdef
#if AOTRITON_V3_API  // if constexpr does not stop errors from undefined functions
      using aotriton::v3::flash::CausalType;
      using aotriton::v3::flash::VarlenType;
      using aotriton::v3::flash::WindowValue;
      aotriton::v3::flash::attn_fwd_params params;
      params.Q = mk_aotensor(q_t, "q");
      params.K = mk_aotensor(k_t, "k");
      params.V = mk_aotensor(v_t, "v");
      params.Sm_scale = softmax_scale;
      params.L = compute_logsumexp ? mk_aotensor<2>(softmax_lse, "M") : empty_t2;
      params.Out = mk_aotensor(output_t, "Out");
      params.Max_seqlen_q = max_seqlen_q;    // Unused if cu_seqlens_q is empty
      params.Max_seqlen_k = max_seqlen_k;    // Unused if cu_seqlens_k is empty
      params.dropout_p = dropout_p;
      params.philox_seed_ptr = seed;
      params.philox_offset1 = offset1;
      params.philox_offset2 = offset2;
      params.philox_seed_output = seed_output;
      params.philox_offset_output = offset_output;
      params.encoded_softmax = mk_aotensor(softmax_fa_t, "encoded_softmax");
      params.persistent_atomic_counter = persistent_counter;
      params.causal_type = is_causal ? CausalType::WindowedAttention : CausalType::None;
      if (static_cast<int64_t>(sdp::CustomMaskType::CausalFromTopLeft) == custom_mask_type) {
        params.window_left = WindowValue::TopLeftAligned;
        params.window_right = WindowValue::TopLeftAligned;
      } else if (static_cast<int64_t>(sdp::CustomMaskType::CausalFromBottomRight) == custom_mask_type) {
        params.window_left = WindowValue::BottomRightAligned;
        params.window_right = WindowValue::BottomRightAligned;
      }
      if (bias.has_value()) {
        params.B = mk_aotensor(bias.value(), "bias");
      }
      if (seqstart_q.has_value()) {
        params.varlen_type = VarlenType::CompactVarlen;
        params.cu_seqlens_q = mk_aotensor<1>(seqstart_q.value(), "cu_seqlens_q");
        params.cu_seqlens_k = mk_aotensor<1>(seqstart_k.value(), "cu_seqlens_k");
      } else {
        params.varlen_type = VarlenType::None;
      }
      err = aotriton::v3::flash::attn_fwd(params,
                                          aotriton::v3::flash::attn_fwd_params::kVersion,
                                          stream);
#endif  // AOTRITON_V3_API
    } else if (seqstart_q.has_value()) {
      // varlen aka nested tensor
      err = attn_fwd_compact_varlen(mk_aotensor(q_t, "q"),
                                    mk_aotensor(k_t, "k"),
                                    mk_aotensor(v_t, "v"),
                                    bias.has_value() ? mk_aotensor(bias.value(), "bias"): empty_t4,
                                    mk_aotensor<1>(seqstart_q.value(), "cu_seqlens_q"),
                                    mk_aotensor<1>(seqstart_k.value(), "cu_seqlens_k"),
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    softmax_scale,
                                    compute_logsumexp ? mk_aotensor<2>(softmax_lse, "M") : empty_t2,
                                    mk_aotensor(output_t, "Out"),
                                    dropout_p,
                                    seed,
                                    offset1,
                                    offset2,
                                    seed_output,
                                    offset_output,
                                    mk_aotensor(softmax_fa_t, "encoded_softmax"),
                                    is_causal,
                                    persistent_counter,
                                    stream);
    } else {
      err = attn_fwd(mk_aotensor(q_t, "q"),
                     mk_aotensor(k_t, "k"),
                     mk_aotensor(v_t, "v"),
                     bias.has_value() ? mk_aotensor(bias.value(), "bias"): empty_t4,
                     softmax_scale,
                     compute_logsumexp ? mk_aotensor<2>(softmax_lse, "M") : empty_t2,
                     mk_aotensor(output_t, "Out"),
                     dropout_p,
                     seed,
                     offset1,
                     offset2,
                     seed_output,
                     offset_output,
                     mk_aotensor(softmax_fa_t, "encoded_softmax"),
                     is_causal,
                     persistent_counter,
                     stream);
    }
#else
    TORCH_CHECK(false, "Attempting to use AOTriton mem_eff_forward backend in a build that has not built AOTriton");
#endif
  } // CK BACKEND
#else
  // CUDA Implementation
  cudaDeviceProp* p = at::cuda::getDeviceProperties(query.device().index());
  int computeCapability = p->major * 10 + p->minor;
  if (computeCapability == 121) {
    computeCapability = 120;
  }

  bool kernel_launched = false;
  const auto maxShmem = p->sharedMemPerBlockOptin;

  auto launchKernel = [&](auto _k, auto kernel_fn) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    if (kernel_launched) {
      return;
    }
    // Check if this kernel is compatible
    if (!Kernel::kSupportsDropout && use_dropout) {
      return;
    }
    if (!Kernel::kSupportsBias && bias.has_value()) {
      return;
    }

    if (value.size(3) > Kernel::kMaxK || key.size(3) > Kernel::kMaxK) {
      return;
    }
    // Alignment
    if ((query.stride(2) % Kernel::kAlignmentQ) ||
        (key.stride(2) % Kernel::kAlignmentK) ||
        (value.stride(2) % Kernel::kAlignmentV)) {
      return;
    }
    // Uses too much shmem
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > maxShmem) {
      return;
    }
    kernel_launched = true;

    res = at::empty(
        {B, M, num_heads, Kv},
        query.options().dtype(
            CutlassToAtenDtype<typename Kernel::output_t>::atScalarType()));

    // NOTE: Should be aligned (by padding) in case M is
    // not a good number for loading during backward
    constexpr decltype(M) kAlignLSE = Kernel::kAlignLSE;
    logsumexp = at::empty(
        {seqstart_q.has_value() ? seqstart_q->size(0) - 1 : B,
         num_heads,
         compute_logsumexp ? ceil_div(max_seqlen_q, kAlignLSE) * kAlignLSE : 0},
        query.options().dtype(at::ScalarType::Float));
    typename Kernel::Params p;
    p.query_ptr = (const scalar_t*)query.const_data_ptr();
    p.key_ptr = (const scalar_t*)key.const_data_ptr();
    p.value_ptr = (const scalar_t*)value.const_data_ptr();
    p.logsumexp_ptr = compute_logsumexp
        ? (typename Kernel::lse_scalar_t*)logsumexp.data_ptr()
        : nullptr;
    at::Tensor output_accum;
    if (Kernel::kNeedsOutputAccumulatorBuffer) {
      output_accum = at::empty(
          {B, M, num_heads, Kv},
          query.options().dtype(
              CutlassToAtenDtype<
                  typename Kernel::output_accum_t>::atScalarType()));
      p.output_accum_ptr =
          (typename Kernel::output_accum_t*)output_accum.data_ptr();
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.output_ptr = (typename Kernel::output_t*)res.data_ptr();

    if (seqstart_q.has_value()) {
      p.seqstart_q_ptr = (const int32_t*)seqstart_q->const_data_ptr();
      p.seqstart_k_ptr = (const int32_t*)seqstart_k->const_data_ptr();
    }

    p.num_heads = num_heads;
    p.head_dim = query.size(3);
    p.head_dim_value = value.size(3);
    p.num_queries = max_seqlen_q;
    p.num_keys = max_seqlen_k;
    p.num_batches = seqstart_q.has_value() ? seqstart_q->size(0) - 1 : B;
    p.custom_mask_type = custom_mask_type;

    p.seqlen_k_ptr = nullptr;
    if (seqlen_k.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(seqlen_k.value());
      TORCH_CHECK(seqlen_k->scalar_type() == at::ScalarType::Int);
      p.seqlen_k_ptr = (const int32_t*)seqlen_k->const_data_ptr();
    }
    if (window_size.has_value()) {
      p.window_size = *window_size;
    }
    p.scale = sdp::calculate_scale(query, scale).expect_float();

    ASSIGN_CHECK_OVERFLOW(p.q_strideB, query.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.k_strideB, key.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.v_strideB, value.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.q_strideM, query.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.k_strideM, key.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.v_strideM, value.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.q_strideH, query.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.k_strideH, key.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.v_strideH, value.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.o_strideM, res.stride(1));

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(
          bias->scalar_type() == CutlassToAtenDtype<scalar_t>::atScalarType(),
          "invalid dtype for bias - should match query's dtype");
      p.attn_bias_ptr = (const scalar_t*)bias->const_data_ptr();

      TORCH_CHECK(bias->dim() == 4, "Bias expected in BMHK format");
      TORCH_CHECK(
          bias->size(0) == query.size(0),
          "attn_bias: wrong shape (batch dimension)");
      TORCH_CHECK(
          bias->size(1) == query.size(2),
          "attn_bias: wrong shape (head dimension)");
      TORCH_CHECK(
          bias->size(2) == query.size(1),
          "attn_bias: wrong shape (seqlenQ dimension)");
      TORCH_CHECK(
          bias->size(3) == key.size(1),
          "attn_bias: wrong shape (seqlenKV dimension)");
      ASSIGN_CHECK_OVERFLOW(p.bias_strideB, bias->stride(0));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideH, bias->stride(1));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideM, bias->stride(2));
      TORCH_CHECK(
          bias->stride(3) == 1,
          "attn_bias: wrong alignment (last dimension must be contiguous)");
    }

    p.use_dropout = use_dropout;
    if (p.use_dropout) {
      p.rng_engine_inputs = philox_state;
      p.dropout_prob = dropout_p;
      p.seed = seed_t.data_ptr<int64_t>();
      p.extragraph_offset = offset_t.data_ptr<int64_t>();
    }

    if (smem_bytes > 0xc000) {
      auto err = cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      TORCH_CHECK(
          err != cudaErrorInvalidValue,
          "This GPU does not have enough shared-memory (kernel requires ",
          smem_bytes / 1024,
          " kb)");
      AT_CUDA_CHECK(err);
    }
    auto blocks = p.getBlocksGrid();
    if (blocks.x * blocks.y * blocks.z == 0 || key.size(1) == 0) {
      res.zero_();
      return;
    }
    Kernel::check_supported(p);
    kernel_fn<<<blocks, p.getThreadsGrid(), smem_bytes, stream>>>(p);
  };

  // Dispatch to the right kernel
  DISPATCH_TYPES(query, ([&]() {
                   dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
                 }));
  TORCH_CHECK(kernel_launched, "cutlassF: no kernel found to launch!");
  AT_CUDA_CHECK(cudaGetLastError());

#endif // USE_ROCM
  return std::make_tuple(
      std::move(res),
      std::move(logsumexp),
      std::move(seed_t),
      std::move(offset_t),
      max_seqlen_q,
      // TODO: why isn't this being set in the kernel?
      max_seqlen_k_.has_value() ? max_seqlen_k_.value() : max_seqlen_k);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mem_efficient_attention_backward_cutlass_origin(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& kernel_bias, // additive attention bias
    const at::Tensor& out,
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const std::optional<at::Tensor>& cu_seqlens_q_dummy,
    // (Mode 1MHK only) [b+1]: cu_seqlens_k[b] contains the
    // position of the first key token for batch $b
    const std::optional<at::Tensor>& cu_seqlens_k_dummy,
    // (Mode 1MHK only) Maximum sequence length across batches
    int64_t max_seqlen_q,
    // (Mode 1MHK only) Maximum sequence length across batches
    int64_t max_seqlen_k,
    const at::Tensor& logsumexp,
    double dropout_p, // dropout probability
    const at::Tensor&
        philox_seed, // seed using for generating random numbers for dropout
    const at::Tensor& philox_offset, // offset into random number sequence
    int64_t custom_mask_type,
    const bool bias_requires_grad,
    const std::optional<double> scale,
    std::optional<int64_t> num_splits_key,
    const std::optional<int64_t> window_size,
    const bool shared_storage_dqdkdv){
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }
  // This path is used when we directly call _efficient_attention_forward
  // from python.
  // This is needed because SaveVariable automatically converts
  // std::optional to undefined tensor
  std::optional<Tensor> bias, cu_seqlens_q, cu_seqlens_k;
  bias = kernel_bias.has_value() && !kernel_bias->defined() ? std::nullopt : kernel_bias;
  cu_seqlens_q = cu_seqlens_q_dummy.has_value() && !cu_seqlens_q_dummy->defined() ? std::nullopt : cu_seqlens_q_dummy;
  cu_seqlens_k = cu_seqlens_k_dummy.has_value() && !cu_seqlens_k_dummy->defined() ? std::nullopt : cu_seqlens_k_dummy;

    // ndim
  TORCH_CHECK(query.dim() == grad_out_.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 4);

  // batch size
  TORCH_CHECK(query.size(0) == grad_out_.size(0));
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // seqlen
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(query.size(1) == grad_out_.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));
  TORCH_CHECK(query.size(2) == grad_out_.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));
  TORCH_CHECK(value.size(3) == grad_out_.size(3));

  // handle potentially non-contiguous grad_out through a copy
  auto grad_out = grad_out_.contiguous();
  CHECK_NOSPARSE_CONTIGUOUS_CUDA(grad_out);

  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  TORCH_CHECK(cu_seqlens_q.has_value() == cu_seqlens_k.has_value());
  TORCH_CHECK(
      !(cu_seqlens_q.has_value() && bias.has_value()),
      "cu seqlen + bias not supported");
  if (cu_seqlens_q.has_value()) {
    TORCH_CHECK(cu_seqlens_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_q->dim() == 1 && cu_seqlens_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*cu_seqlens_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*cu_seqlens_k));
    TORCH_CHECK(cu_seqlens_q->size(0) == cu_seqlens_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q > 0, "max_seqlen_q required with `cu_seqlens_q`");
    TORCH_CHECK(max_seqlen_k > 0, "max_seqlen_k required with `cu_seqlens_k`");
    TORCH_CHECK(
        max_seqlen_k <= key.size(1), "Invalid max_seqlen_k:", max_seqlen_k);
    TORCH_CHECK(
        max_seqlen_q <= query.size(1), "Invalid max_seqlen_q:", max_seqlen_q);
  } else {
    max_seqlen_q = query.size(1);
    max_seqlen_k = key.size(1);
  }

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t nH = query.size(2);
  int64_t K = query.size(3);
  int64_t Kv = value.size(3);

  at::Tensor grad_q, grad_k, grad_v, grad_bias;
  if (shared_storage_dqdkdv) {
    // Create one big contiguous chunk
    // This is because q, k and v usually come from a single
    // output of a linear layer that is chunked.
    // Creating the gradients with the right layout saves us
    // a `torch.cat` call in the backward pass
    TORCH_CHECK(
      query.size(1) == key.size(1),
      "`shared_storage_dqdkdv` is only supported when Q/K/V "
      "have the same sequence length: got ", query.size(1),
      " query tokens and ", key.size(1), " key/value tokens"
    );
    TORCH_CHECK(
      query.size(3) == key.size(3),
      "`shared_storage_dqdkdv` is only supported when Q/K/V "
      "have the same embed dim: got ", query.size(3),
      " for Q, and ", key.size(3), " for K"
    );
    at::Tensor chunk = at::empty({B, M, 3, nH, K}, query.options());
    grad_q = chunk.select(2, 0);
    grad_k = chunk.select(2, 1);
    grad_v = chunk.select(2, 2);
  } else {
    grad_q = at::empty(query.sizes(), query.options());
    grad_k = at::empty(key.sizes(), key.options());
    grad_v = at::empty(value.sizes(), value.options());
  }

  if (bias_requires_grad) {
    TORCH_CHECK(
        bias.has_value(),
        "bias_requires_grad is true but no bias was provided");
    // force alignment for the last dim
    std::vector<int64_t> sz = bias->sizes().vec();
    int64_t lastDim = sz[sz.size() - 1];
    int64_t alignTo = 16;
    sz[sz.size() - 1] = alignTo * ((lastDim + alignTo - 1) / alignTo);
    grad_bias = at::empty(sz, bias->options())
                    .slice(/*dim=*/-1, /*start=*/0, /*end=*/lastDim);
  }

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;

  // See Note [Seed and Offset Device]
  at::PhiloxCudaState rng_engine_inputs;
  if (use_dropout) {
    if (at::cuda::currentStreamCaptureStatus() ==
        at::cuda::CaptureStatus::None) {
      rng_engine_inputs = at::PhiloxCudaState(
          *philox_seed.data_ptr<int64_t>(),
          *philox_offset.data_ptr<int64_t>());
    } else { // dropout + capture
      rng_engine_inputs = at::PhiloxCudaState(
          philox_seed.data_ptr<int64_t>(),
          philox_offset.data_ptr<int64_t>(),
          0);
    }
  }

#ifdef USE_ROCM
  // ROCM Implementation
  if(at::globalContext().getROCmFAPreferredBackend() == at::ROCmFABackend::Ck)
  {
#if defined(USE_ROCM_CK_SDPA)
    const auto my_softmax_scale = sdp::calculate_scale(query, scale).expect_float();
    // Store grad_bias in optional
    std::optional<at::Tensor> opt_grad_bias = grad_bias;
    auto
        [dQ,
         dK,
         dV,
         dBias] =
             pytorch_flash::mem_eff_backward_ck(
                     grad_out,
                     query,
                     key,
                     value,
                     out,
                     logsumexp,
                     grad_q,
                     grad_k,
                     grad_v,
                     bias,
                     bias_requires_grad,
                     opt_grad_bias,
                     cu_seqlens_q,
                     cu_seqlens_k,
                     max_seqlen_q,
                     max_seqlen_k,
                     float(dropout_p),
                     my_softmax_scale,
                     custom_mask_type == 0 ? false : true, // is_causal
                     false, // deterministic
                     false, // zero_tensors
                     philox_seed,
                     philox_offset);
    grad_bias = dBias;
#else
    TORCH_CHECK(false, "Attempting to use CK mem_eff_backward backend in a build that has not built CK");
#endif
  } else {
#ifndef DISABLE_AOTRITON
    TORCH_CHECK(!num_splits_key.has_value(),
              "ROCM does not support num_split_keys in _efficient_attention_forward");
    TORCH_CHECK(!window_size.has_value(),
              "ROCM does not support window_size in _efficient_attention_forward");
    auto ret = aotriton::v2::flash::check_gpu(stream);
    if (hipSuccess != ret) {
      TORCH_CHECK(false,
                "[AOTriton] Accelerated SDPA only supports MI200/MI300X/7900XTX/9070XT GPUs"
                " (gfx90a/gfx942/gfx1100/gfx1201)")
    }
    const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
    bool is_causal;
    if (static_cast<int64_t>(sdp::CustomMaskType::NoCustomMask) == custom_mask_type) {
      is_causal = false;
    } else {
      is_causal = true;
#if AOTRITON_V3_API == 0
      if (static_cast<int64_t>(sdp::CustomMaskType::CausalFromTopLeft) != custom_mask_type) {
        TORCH_CHECK(false, "[_efficient_attention_forward] Unsupported mask type on ROCM, for now");
      }
#endif
    }
    at::Tensor q_t = query.permute({0,2,1,3});
    at::Tensor k_t = key.permute({0,2,1,3});
    at::Tensor v_t = value.permute({0,2,1,3});
    at::Tensor out_t = out.permute({0,2,1,3});
    at::Tensor dq_t = grad_q.permute({0,2,1,3});
    at::Tensor dk_t = grad_k.permute({0,2,1,3});
    at::Tensor dv_t = grad_v.permute({0,2,1,3});
    at::Tensor dout_t = grad_out.permute({0,2,1,3});
    at::Tensor softmax_lse = logsumexp.view({B * nH, max_seqlen_q});
    hipError_t err;
    using aotriton::v2::flash::attn_bwd;
    using aotriton::v2::flash::attn_bwd_fused;
    using aotriton::v2::flash::attn_bwd_compact_varlen;
    using sdp::aotriton_adapter::mk_aotensor;
    using sdp::aotriton_adapter::mk_aoscalartensor;
    using sdp::aotriton_adapter::cast_dtype;
    aotriton::TensorView<4> empty_t4(0, {0, 0, 0, 0}, {0, 0, 0, 0}, cast_dtype(query.dtype()));
    if constexpr (AOTRITON_ALWAYS_V3_API) {  // Better readability than nesting ifdef
#if AOTRITON_V3_API  // if constexpr does not stop errors from undefined functions
      using aotriton::v3::flash::CausalType;
      using aotriton::v3::flash::VarlenType;
      using aotriton::v3::flash::WindowValue;
      aotriton::v3::flash::attn_bwd_params params;
      params.Q = mk_aotensor(q_t, "q");
      params.K = mk_aotensor(k_t, "k");
      params.V = mk_aotensor(v_t, "v");
      params.B = bias.has_value() ? mk_aotensor(bias.value(), "bias") : empty_t4;
      params.Sm_scale = softmax_scale;
      params.Out = mk_aotensor(out_t, "out");
      params.DO = mk_aotensor(dout_t, "dout");
      params.DK = mk_aotensor(dk_t, "dk");
      params.DV = mk_aotensor(dv_t, "dv");
      params.DQ = mk_aotensor(dq_t, "dq");
      params.DB = bias_requires_grad ? mk_aotensor(grad_bias, "db") : empty_t4;
      params.L = mk_aotensor<2>(softmax_lse, "L");
      params.Max_seqlen_q = max_seqlen_q;        // Unused if cu_seqlens_q is empty
      params.Max_seqlen_k = max_seqlen_k;        // Unused if cu_seqlens_k is empty
      params.dropout_p = float(dropout_p);
      params.philox_seed_ptr =  mk_aoscalartensor(philox_seed);
      params.philox_offset1 = mk_aoscalartensor(philox_offset);
      params.philox_offset2 = 0;
      params.causal_type = is_causal ? CausalType::WindowedAttention : CausalType::None;
      if (static_cast<int64_t>(sdp::CustomMaskType::CausalFromTopLeft) == custom_mask_type) {
        params.window_left = WindowValue::TopLeftAligned;
        params.window_right = WindowValue::TopLeftAligned;
      } else if (static_cast<int64_t>(sdp::CustomMaskType::CausalFromBottomRight) == custom_mask_type) {
        params.window_left = WindowValue::BottomRightAligned;
        params.window_right = WindowValue::BottomRightAligned;
      }
#if AOTRITON_ALWAYS_V3_API
      using sdp::aotriton_adapter::mklazy_empty_like;
      using sdp::aotriton_adapter::mklazy_fp32zeros;
      using sdp::aotriton_adapter::LazyTensorContext;
      LazyTensorContext lazy_delta { .like_tensor = softmax_lse, .tensor_name = "delta" };
      LazyTensorContext lazy_dq_acc { .like_tensor = dq_t, .tensor_name = "dq_acc" };
      params.D = mklazy_empty_like<2>(&lazy_delta);
      params.DQ_ACC = mklazy_fp32zeros<4>(&lazy_dq_acc);
#else
      at::Tensor delta = at::empty_like(softmax_lse).contiguous();
      params.D = mk_aotensor<2>(delta, "delta");
#endif
      if (cu_seqlens_q.has_value()) {
        params.varlen_type = VarlenType::CompactVarlen;
        params.cu_seqlens_q = mk_aotensor<1>(cu_seqlens_q.value(), "cu_seqlens_q");
        params.cu_seqlens_k = mk_aotensor<1>(cu_seqlens_k.value(), "cu_seqlens_k");
      } else {
        params.varlen_type = VarlenType::None;
      }
      err = aotriton::v3::flash::attn_bwd(params,
                                          aotriton::v3::flash::attn_bwd_params::kVersion,
                                          stream);
#endif  // AOTRITON_V3_API
    } else if (cu_seqlens_q.has_value()) {
      at::Tensor delta = at::empty_like(softmax_lse).contiguous();
      // varlen aka Nested tensor
      err = attn_bwd_compact_varlen(mk_aotensor(q_t, "q"),
                                    mk_aotensor(k_t, "k"),
                                    mk_aotensor(v_t, "v"),
                                    mk_aotensor<1>(cu_seqlens_q.value(), "cu_seqlens_q"),
                                    mk_aotensor<1>(cu_seqlens_k.value(), "cu_seqlens_k"),
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    bias.has_value() ? mk_aotensor(bias.value(), "bias") : empty_t4,
                                    softmax_scale,
                                    mk_aotensor(out_t, "out"),
                                    mk_aotensor(dout_t, "dout"),
                                    mk_aotensor(dq_t, "dq"),
                                    mk_aotensor(dk_t, "dk"),
                                    mk_aotensor(dv_t, "dv"),
                                    bias_requires_grad ? mk_aotensor(grad_bias, "db") : empty_t4,
                                    mk_aotensor<2>(softmax_lse, "L"),
                                    mk_aotensor<2>(delta, "delta"),
                                    float(dropout_p),
                                    mk_aoscalartensor(philox_seed),
                                    mk_aoscalartensor(philox_offset),
                                    0,
                                    is_causal,
                                    stream);
    } else { // cu_seqlens.has_value
      auto d_head = Kv;
      bool use_fused_bwd = d_head <= 192 && d_head * max_seqlen_q < 64 * 512;
      if (use_fused_bwd) {
        err = attn_bwd_fused(mk_aotensor(q_t, "q"),
                             mk_aotensor(k_t, "k"),
                             mk_aotensor(v_t, "v"),
                             bias.has_value() ? mk_aotensor(bias.value(), "bias") : empty_t4,
                             softmax_scale,
                             mk_aotensor(out_t, "out"),
                             mk_aotensor(dout_t, "dout"),
                             mk_aotensor(dq_t, "dq"),
                             mk_aotensor(dk_t, "dk"),
                             mk_aotensor(dv_t, "dv"),
                             bias_requires_grad ? mk_aotensor(grad_bias, "db") : empty_t4,
                             mk_aotensor<2>(softmax_lse, "L"),
                             float(dropout_p),
                             mk_aoscalartensor(philox_seed),
                             mk_aoscalartensor(philox_offset),
                             0,
                             is_causal,
                             stream);
      } else {
        at::Tensor delta = at::empty_like(softmax_lse).contiguous();
        err = attn_bwd(mk_aotensor(q_t, "q"),
                     mk_aotensor(k_t, "k"),
                     mk_aotensor(v_t, "v"),
                     bias.has_value() ? mk_aotensor(bias.value(), "bias") : empty_t4,
                     softmax_scale,
                     mk_aotensor(out_t, "out"),
                     mk_aotensor(dout_t, "dout"),
                     mk_aotensor(dq_t, "dq"),
                     mk_aotensor(dk_t, "dk"),
                     mk_aotensor(dv_t, "dv"),
                     bias_requires_grad ? mk_aotensor(grad_bias, "db") : empty_t4,
                     mk_aotensor<2>(softmax_lse, "L"),
                     mk_aotensor<2>(delta, "delta"),
                     float(dropout_p),
                     mk_aoscalartensor(philox_seed),
                     mk_aoscalartensor(philox_offset),
                     0,
                     is_causal,
                     stream);
      } //used_fused_bwd
    } // cuseqlen.has_value
#else  // DISABLE_AOTRITON
    TORCH_CHECK(false, "Attempting to use aotriton mem_eff_backward backend in a build that has not built AOTriton");
#endif
  } // Use CK
#else // USE_CUDA
  at::Tensor workspace;
  cudaDeviceProp* p = at::cuda::getDeviceProperties(query.device().index());
  int computeCapability = p->major * 10 + p->minor;
  if (computeCapability == 121) {
    computeCapability = 120;
  }

  bool kernel_launched = false;
  const auto maxK = std::max(query.size(3), value.size(3));
  const auto maxShmem = p->sharedMemPerBlockOptin;

  auto launchKernel = [&](auto _k, auto kernel_fn) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    if (kernel_launched) {
      return;
    }
    // Check if this kernel is compatible
    if (Kernel::kMaxK < maxK) {
      return;
    }
    // Dropout must be supported if we need it
    if (use_dropout && !Kernel::kApplyDropout) {
      return;
    }
    if (Kernel::kKeysQueriesAlignedToBlockSize &&
        (cu_seqlens_q.has_value() || M % Kernel::kBlockSizeI ||
         N % Kernel::kBlockSizeJ)) {
      return;
    }
    // Alignment
    if ((query.stride(2) % Kernel::kMinimumAlignment) ||
        (key.stride(2) % Kernel::kMinimumAlignment) ||
        (value.stride(2) % Kernel::kMinimumAlignment)) {
      return;
    }
    // Uses too much shmem
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > maxShmem) {
      return;
    }

    kernel_launched = true;

    // TODO: Fuse this into a kernel?
    // This is a bottleneck for smaller sequences (M <= 128)
    auto delta = Kernel::kKernelComputesDelta
        ? at::empty({B, nH, M}, query.options().dtype(at::ScalarType::Float))
        : (grad_out.to(at::kFloat) * out.to(at::kFloat))
              .sum(-1)
              .transpose(-2, -1)
              .contiguous();
    TORCH_INTERNAL_ASSERT(delta.size(0) == B);
    TORCH_INTERNAL_ASSERT(delta.size(1) == nH);
    TORCH_INTERNAL_ASSERT(delta.size(2) == M);

    typename Kernel::Params p;
    p.query_ptr = (const scalar_t*)query.const_data_ptr();
    p.key_ptr = (const scalar_t*)key.const_data_ptr();
    p.value_ptr = (const scalar_t*)value.const_data_ptr();
    p.logsumexp_ptr = (typename Kernel::lse_scalar_t const *)logsumexp.const_data_ptr();
    p.output_ptr = (const scalar_t*)out.const_data_ptr();
    p.grad_output_ptr = (const scalar_t*)grad_out.const_data_ptr();
    p.grad_query_ptr = (scalar_t*)grad_q.data_ptr();
    p.grad_key_ptr = (scalar_t*)grad_k.data_ptr();
    p.grad_value_ptr = (scalar_t*)grad_v.data_ptr();
    p.delta_ptr = (float*)delta.data_ptr();
    p.head_dim = query.size(3);
    p.head_dim_value = value.size(3);
    p.num_queries = max_seqlen_q;
    p.num_keys = max_seqlen_k;
    p.num_batches = cu_seqlens_q.has_value() ? cu_seqlens_q->size(0) - 1 : B;
    p.num_heads = nH;
    p.custom_mask_type = custom_mask_type;
    p.scale = sdp::calculate_scale(query, scale).expect_float();
    if (cu_seqlens_q.has_value()) {
      p.cu_seqlens_q_ptr = (const int32_t*)cu_seqlens_q->const_data_ptr();
      p.cu_seqlens_k_ptr = (const int32_t*)cu_seqlens_k->const_data_ptr();
    }
    if (window_size.has_value()) {
      p.window_size = *window_size;
    }

    ASSIGN_CHECK_OVERFLOW(p.lse_strideB, logsumexp.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.lse_strideH, logsumexp.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.gO_strideB, grad_out.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gO_strideM, grad_out.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.gO_strideH, grad_out.stride(2));

    ASSIGN_CHECK_OVERFLOW(p.o_strideB, out.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.o_strideH, out.stride(2));

    ASSIGN_CHECK_OVERFLOW(p.gQ_strideB, grad_q.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gK_strideB, grad_k.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gV_strideB, grad_v.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gQ_strideH, grad_q.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.gK_strideH, grad_k.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.gV_strideH, grad_v.stride(2));
    p.gQKV_strideM_multiplier = shared_storage_dqdkdv ? 3 : 1;
    TORCH_INTERNAL_ASSERT(p.gQ_strideM() == grad_q.stride(1));
    TORCH_INTERNAL_ASSERT(p.gK_strideM() == grad_k.stride(1));
    TORCH_INTERNAL_ASSERT(p.gV_strideM() == grad_v.stride(1));

    ASSIGN_CHECK_OVERFLOW(p.q_strideB, query.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.k_strideB, key.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.v_strideB, value.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.q_strideM, query.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.k_strideM, key.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.v_strideM, value.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.q_strideH, query.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.k_strideH, key.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.v_strideH, value.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.delta_strideB, delta.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.delta_strideH, delta.stride(1));

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(
          bias->scalar_type() == CutlassToAtenDtype<scalar_t>::atScalarType(),
          "invalid dtype for bias - should match query's dtype");

      p.bias_ptr = (scalar_t*)bias->data_ptr();

      TORCH_CHECK(bias->dim() == 4, "Bias expected in BMHK format");
      TORCH_CHECK(
          bias->size(0) == query.size(0),
          "attn_bias: wrong shape (batch dimension)");
      TORCH_CHECK(
          bias->size(1) == query.size(2),
          "attn_bias: wrong shape (head dimension)");
      TORCH_CHECK(
          bias->size(2) == query.size(1),
          "attn_bias: wrong shape (seqlenQ dimension)");
      TORCH_CHECK(
          bias->size(3) == key.size(1),
          "attn_bias: wrong shape (seqlenKV dimension)");
      TORCH_CHECK(
          bias->stride(3) == 1,
          "attn_bias: wrong alignment (last dimension must be contiguous)");
      ASSIGN_CHECK_OVERFLOW(p.bias_strideB, bias->stride(0));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideH, bias->stride(1));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideM, bias->stride(2));

      if (bias_requires_grad) {
        p.grad_bias_ptr = (scalar_t*)grad_bias.data_ptr();

        ASSIGN_CHECK_OVERFLOW(p.gB_strideB, grad_bias.stride(0));
        ASSIGN_CHECK_OVERFLOW(p.gB_strideH, grad_bias.stride(1));
        ASSIGN_CHECK_OVERFLOW(p.gB_strideM, grad_bias.stride(2));
      }
    }

    if (use_dropout) {
      p.rng_engine_inputs = rng_engine_inputs;
      p.dropout_prob = dropout_p;
    }

    // Heuristic for finding optimal number of splits
    auto parallelism_without_split_key =
        p.getBlocksGrid().x * p.getBlocksGrid().y * p.getBlocksGrid().z;
    p.num_splits_key = cutlass::ceil_div(p.num_keys, Kernel::kBlockSizeJ);
    if (num_splits_key.has_value()) {
      p.num_splits_key =
          std::min<int64_t>(p.num_splits_key, num_splits_key.value());
    } else {
      // Keys splitting heuristic

      // If we already have enough parallelism, split-keys can help
      // better use L2 cache.
      // This is negligible when the seqlen is too small tho
      if (parallelism_without_split_key >= 256 &&
          p.num_keys <= 2 * Kernel::kBlockSizeJ) {
        p.num_splits_key = 1;
      }
      // Increasing `split_keys` leads to using more gmem for temporary storage
      // when we need a staging area for gK/gV. let's avoid that
      if (Kernel::kNeedsAccumGradK || Kernel::kNeedsAccumGradV) {
        p.num_splits_key = std::min(
            int32_t(p.num_splits_key), 200 / ((int32_t)(p.num_batches * p.num_heads)));
      }
    }
    if (!Kernel::kEnableSplitKeys || p.num_splits_key < 1) {
      p.num_splits_key = 1;
    }

    auto& ctx = at::globalContext();
    if (ctx.deterministicAlgorithms()) {
      if (ctx.deterministicAlgorithmsWarnOnly()) {
        TORCH_WARN_ONCE(
            "Memory Efficient attention defaults to a non-deterministic algorithm. ",
            "To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).");
      } else {
        TORCH_CHECK(
            num_splits_key.value_or(1) <= 1,
            "Using `num_splits_key > 1` makes the algorithm non-deterministic, and pytorch's deterministic mode is enabled");
        p.num_splits_key = 1;
      }
    }
    int64_t size_bytes = p.workspace_size();
    if (size_bytes) {
      workspace =
          at::empty({size_bytes}, query.options().dtype(at::ScalarType::Byte));
      p.workspace = (float*)workspace.data_ptr();
      if (p.should_zero_workspace()) {
        workspace.zero_();
      }
    }

    // Handle the edge-cases where some tensors are empty
    if (p.num_queries == 0 || p.num_keys == 0 || p.num_batches == 0 ||
        p.num_heads == 0) {
      grad_k.zero_();
      grad_v.zero_();
      grad_q.zero_();
      return;
    }
    Kernel::check_supported(p);

    if (smem_bytes > 0xc000) {
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
      auto err = cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      TORCH_CHECK(
          err != cudaErrorInvalidValue,
          "This GPU does not have enough shared-memory (kernel requires ",
          smem_bytes / 1024,
          " kb)");
      AT_CUDA_CHECK(err);
    }

    // second syntax resulted in the error below on windows
    // error C3495: 'kernel_fn': a simple capture must be a variable
    // with automatic storage duration declared
    // in the reaching scope of the lambda
#ifdef _WIN32
    cudaFuncAttributes attr;
    AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
    TORCH_INTERNAL_ASSERT(
        attr.binaryVersion >= Kernel::ArchTag::kMinComputeCapability,
        "Something went wrong in the build process");
#else
    auto checkBinaryArchMatches = [&]() {
      cudaFuncAttributes attr;
      AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
      return attr.binaryVersion >= Kernel::ArchTag::kMinComputeCapability;
    };
    TORCH_INTERNAL_ASSERT(
        checkBinaryArchMatches(), "Something went wrong in the build process");
#endif

    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
  };

  DISPATCH_TYPES(query, ([&]() {
                   dispatch_cutlassB<scalar_t>(launchKernel, computeCapability);
                 }));
  TORCH_CHECK(kernel_launched, "cutlassB: no kernel found to launch!");
  AT_CUDA_CHECK(cudaGetLastError());
#endif // USE_ROCM
  return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v), std::move(grad_bias));
}
}   //namespace native
}   //namespace at