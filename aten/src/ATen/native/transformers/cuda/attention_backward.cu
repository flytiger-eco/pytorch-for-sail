#include <string_view>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <cstdint>
#include <type_traits>

#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/Exception.h>
#include <c10/util/bit_cast.h>

#include <c10/core/TensorImpl.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/_cudnn_attention_backward.h>
#include <ATen/ops/_cudnn_attention_backward_native.h>
#include <ATen/ops/_flash_attention_backward.h>
#include <ATen/ops/_flash_attention_backward_native.h>
#include <ATen/ops/_efficient_attention_backward.h>
#include <ATen/ops/_efficient_attention_backward_native.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_backward_native.h>
#endif

#ifdef USE_FLASH_ATTENTION
// FlashAttention Specific Imports
#include <ATen/native/transformers/cuda/flash_attn/flash_api.h>
#endif
#ifdef USE_MEM_EFF_ATTENTION
#ifndef USE_ROCM
// MemoryEfficient Attention Specific Imports for CUDA
#ifdef USE_PPU
#include <xformers/csrc/attention/cuda/fmha/mem_eff_api.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/mem_eff_api.h>
#endif // USE_PPU
#else
#include <ATen/native/transformers/hip/gemm_kernel_utils.h>
// MemoryEfficient Attention Specific Imports for ROCM
#ifndef DISABLE_AOTRITON
#include <ATen/native/transformers/hip/aotriton_adapter.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#endif
#include <ATen/native/transformers/hip/flash_attn/ck/me_ck_api.h>
#endif
#endif

#ifdef __HIP_PLATFORM_AMD__
#include <ATen/native/cudnn/hip/MHA.h>
#else
#include <ATen/native/cudnn/MHA.h>
#endif

#include <cuda_runtime.h>
#include <cuda.h>

namespace at::native {

std::string getDeviceArchitecture() {
  int deviceID = 0;
  cudaError_t err = cudaGetDevice(&deviceID);
  TORCH_CHECK(err == cudaSuccess, "Error getting current device ID: ", cudaGetErrorString(err));
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, deviceID);
  TORCH_CHECK(err == cudaSuccess, "Error getting device properties: ", cudaGetErrorString(err));
  return std::to_string(prop.major) + "." + std::to_string(prop.minor);
}

std::tuple<Tensor, Tensor, Tensor> _flash_attention_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right) {
#if defined(USE_FLASH_ATTENTION)
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  //  CUDA code assumes that dout is contiguous
  auto contiguous_grad_out = grad_out.contiguous();
  auto contiguous_out = out.contiguous();

#ifdef USE_ROCM  // ROCM backend accepts std::optional for window_size_left/right directly.
#ifdef DISABLE_AOTRITON  // CK backend, Passing window_size as it is
  const auto window_left = window_size_left;
  const auto window_right = window_size_right;
#else  // AOTriton implements "generalized" SWA and negative size means negative shifting.
  // aotriton_adapter::parse_window_size tries to match the behavior of CUTLASS backend
  using sdp::aotriton_adapter::parse_window_size;
  const auto [window_left, window_right] = parse_window_size(window_size_left,
                                                             window_size_right);
#endif
#else  // USE_ROCM
  const int window_left = window_size_left.value_or(-1);
  const int window_right = window_size_right.value_or(-1);
#endif  // USE_ROCM

  std::optional<at::Tensor> dq{std::nullopt};
  std::optional<at::Tensor> dk{std::nullopt};
  std::optional<at::Tensor> dv{std::nullopt};

  //  The kernel computes regardless we will drop for this functions return
  Tensor grad_softmax;

  // Currently unused args:
  std::optional<at::Tensor> alibi_slopes{std::nullopt};
  const float softcap = 0.0;

  bool deterministic{false};
  auto& ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (ctx.deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN_ONCE(
          "Flash Attention defaults to a non-deterministic algorithm. ",
          "To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).");
    } else {
      deterministic = true;
    }
  }

  // We check the whether the cumulative_sequence_length_q is defined
  // in order to determine whether we are using varlen or dense forward
  if (cumulative_sequence_length_q.defined()) {
    // Varlen forward
    auto [dQuery, dKey, dValue, dSoftmax] = FLASH_NAMESPACE::mha_varlen_bwd(
        contiguous_grad_out,
        query,
        key,
        value,
        contiguous_out,
        logsumexp,
        dq,
        dk,
        dv,
        cumulative_sequence_length_q,
        cumulative_sequence_length_k,
        alibi_slopes,
        max_seqlen_batch_q,
        max_seqlen_batch_k,
        dropout_p,
        softmax_scale,
        false /*zero_tensors*/,
        is_causal,
        window_left,
        window_right,
        softcap,
        deterministic,
        philox_seed,
        philox_offset);
    return std::make_tuple(std::move(dQuery), std::move(dKey), std::move(dValue));
  } else {
    // Dense forward
    auto [dQuery, dKey, dValue, dSoftmax] = FLASH_NAMESPACE::mha_bwd(
        contiguous_grad_out,
        query,
        key,
        value,
        contiguous_out,
        logsumexp,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        is_causal,
        window_left,
        window_right,
        softcap,
        deterministic,
        philox_seed,
        philox_offset);
    return std::make_tuple(std::move(dQuery), std::move(dKey), std::move(dValue));
  }
#endif
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.");
  return std::make_tuple(Tensor(), Tensor(), Tensor());
}

std::tuple<Tensor, Tensor, Tensor> _cudnn_attention_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    const Tensor& attn_bias,
    const Tensor& cum_seq_q,
    const Tensor& cum_seq_k,
    const int64_t max_q,
    const int64_t max_k,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {

    auto& ctx = at::globalContext();
    if (ctx.deterministicAlgorithms()) {
      if (ctx.deterministicAlgorithmsWarnOnly()) {
        TORCH_WARN_ONCE(
            "cuDNN Attention defaults to a non-deterministic algorithm. ",
            "To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).");
      }
    }

    const bool is_nested = cum_seq_q.defined();
    const int64_t max_seqlen_batch_q = query.size(2);
    const int64_t max_seqlen_batch_k = key.size(2);

    if (!is_nested) {
      const int64_t batch_size = query.size(0);
      const int64_t num_heads = query.size(1);
      const int64_t head_dim_qk = query.size(3);
      const int64_t head_dim_v = value.size(3);

      // This is needed because SaveVariable automatically converts
      // std::optional to undefined tensor
      std::optional<Tensor> attn_bias_;
      if (attn_bias.defined()) {
        attn_bias_ = attn_bias;
      }
      if (attn_bias_.has_value()) {
        const auto bias_dim = attn_bias_.value().dim();
        if (bias_dim == 2) {
          attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_k});
        } else if (bias_dim == 3) {
          attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_k});
        } else {
          TORCH_CHECK(bias_dim == 4, "cuDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got ", attn_bias_.value().dim(), "D");
          attn_bias_ = attn_bias_.value().expand({batch_size, attn_bias_.value().size(1), max_seqlen_batch_q, max_seqlen_batch_k});
        }
      }

      const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
      auto dq = at::empty_like(query);
      auto dk = at::empty_like(key);
      auto dv = at::empty_like(value);
      run_cudnn_SDP_bprop(batch_size /*int64_t b*/,
                          num_heads /*int64_t h*/,
                          max_q/*int64_t s_q*/,
                          max_k/*int64_t s_kv*/,
                          head_dim_qk /*int64_t d_qk*/,
                          head_dim_v /*int64_t d_v*/,
                          softmax_scale /*float scaling_factor*/,
                          is_causal /*bool is_causal*/,
                          dropout_p /*float dropout_probability*/,
                          query /*const Tensor& q*/,
                          key /*const Tensor& k*/,
                          value /*const Tensor& v*/,
                          attn_bias_ /*const std::optional<Tensor>& attn_bias*/,
                          out /*const Tensor& o*/,
                          grad_out/*const Tensor& dO*/,
                          logsumexp/*const Tensor& softmaxstats*/,
                          dq/*Tensor& dQ*/,
                          dk/*Tensor& dK*/,
                          dv/*Tensor& dV*/,
                          philox_seed/*Tensor& dropoutseed*/,
                          philox_offset/*Tensor& dropoutoffset*/);
      return std::make_tuple(std::move(dq), std::move(dk), std::move(dv));
    } else {
      // BHSD ...
      const int64_t batch_size = cum_seq_q.size(0) - 1;
      const int64_t num_heads_q = query.size(-2);
      const int64_t num_heads_k = key.size(-2);
      const int64_t num_heads_v = value.size(-2);
      const int64_t head_dim_qk = query.size(-1);
      const int64_t head_dim_v = value.size(-1);
      std::optional<Tensor> attn_bias_;
      if (attn_bias.defined()) {
        attn_bias_ = attn_bias;
      }
      if (attn_bias_.has_value()) {
        const auto bias_dim = attn_bias_.value().dim();
        if (bias_dim == 2) {
          attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_k});
        } else if (bias_dim == 3) {
          attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_k});
        } else {
          attn_bias_ = attn_bias_.value().expand({batch_size, attn_bias_.value().size(1), max_seqlen_batch_q, max_seqlen_batch_k});
          TORCH_CHECK(bias_dim == 4, "cuDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got ", attn_bias_.value().dim(), "D");
        }
      }

      auto dq = at::empty_like(query);
      auto dk = at::empty_like(key);
      auto dv = at::empty_like(value);

      const auto softmax_scale = sdp::calculate_scale(query, scale).as_float_unchecked();
      run_cudnn_SDP_bprop_nestedtensor(
        batch_size,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        max_seqlen_batch_q,
        max_seqlen_batch_k,
        head_dim_qk,
        head_dim_v,
        softmax_scale,
        is_causal,
        dropout_p,
        cum_seq_q,
        cum_seq_k,
        query,
        key,
        value,
        attn_bias_,
        out,
        grad_out,
        logsumexp,
        dq,
        dk,
        dv,
        philox_seed,
        philox_offset);
      return std::make_tuple(std::move(dq), std::move(dk), std::move(dv));
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_efficient_attention_backward(
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
    const at::Tensor& philox_seed, // seed using for generating random numbers for dropout
    const at::Tensor& philox_offset, // offset into random number sequence
    int64_t custom_mask_type,
    const bool bias_requires_grad,
    const std::optional<double> scale,
    std::optional <int64_t> num_splits_key,
    const std::optional<int64_t> window_size,
    const bool shared_storage_dqdkdv) {
#if defined(USE_MEM_EFF_ATTENTION)
#ifdef USE_PPU
  // Get device arch
  std::string arch = getDeviceArchitecture();
  TORCH_CHECK(arch != "Error", "getDeviceArchitecture failed!");
  // 1. Abstract for PPU1.0 and PPU1.5 path
  // 2. PPU1.0 arch: SM80; PPU1.5 arch: SM89
  if (arch == "8.0") {
    // 1. see https://github.com/pytorch/pytorch/commit/9bd6d6e8b02ec1c6285b6ee785e38ec86ce2f1bd
    // pytorch 2.4 update xformers impl here, add window size param for sliding window and shared_storage_dqdkdv param for saving torch.cat usage
    // in PPU not support this feature for now since the xformers embedded here does not update if not necessary
    // Warning will be raised if windows_size/shared_storage_dqdkdv has value for better debug in the future.
    // 2. since window_size is optional params in PPU impl, we don't pass this here.
    char *pEnv_perf = std::getenv("PPU_SDPA_BACKEND_MEM_EFFI_CE");
    if (pEnv_perf) {
      return mem_efficient_attention_backward_cutlass_origin(
        grad_out_, query, key, value, kernel_bias, out,
        cu_seqlens_q_dummy, cu_seqlens_k_dummy, max_seqlen_q,
        max_seqlen_k, logsumexp, dropout_p,
        philox_seed, philox_offset, custom_mask_type,
        bias_requires_grad, scale, num_splits_key.value_or(0), window_size, shared_storage_dqdkdv);
    }

    if (window_size.has_value()) {
      TORCH_WARN_ONCE("Warning! window_size was used here!");
    }
    if (shared_storage_dqdkdv == true) {
      TORCH_WARN_ONCE("Warning! shared_storage_dqdkdv was used here!");
    }
    // Abstract for PPU1.0 path
    // return [grad_q, grad_k, grad_v, grad_bias]
    return mem_efficient_attention_backward_cutlass(
      grad_out_, query, key, value, kernel_bias, out,
      cu_seqlens_q_dummy, cu_seqlens_k_dummy, max_seqlen_q,
      max_seqlen_k, logsumexp, dropout_p,
      philox_seed, philox_offset, custom_mask_type,
      bias_requires_grad, scale, num_splits_key.value_or(0), window_size);
  } else if (arch == "8.9") {
    // Abstract for PPU1.5 path
    return mem_efficient_attention_backward_cutlass_origin(
      grad_out_, query, key, value, kernel_bias, out,
      cu_seqlens_q_dummy, cu_seqlens_k_dummy, max_seqlen_q,
      max_seqlen_k, logsumexp, dropout_p,
      philox_seed, philox_offset, custom_mask_type,
      bias_requires_grad, scale, num_splits_key.value_or(0), window_size, shared_storage_dqdkdv);
  } else {
    // Not support Arch
    TORCH_CHECK(false, "Unsupported architecture: ", arch, ". Only SM80 (8.0) and SM89 (8.9) are supported.");
  }
#else
  // Abstract for community open source code
  return mem_efficient_attention_backward_cutlass_origin(
    grad_out_, query, key, value, kernel_bias, out,
    cu_seqlens_q_dummy, cu_seqlens_k_dummy, max_seqlen_q,
    max_seqlen_k, logsumexp, dropout_p,
    philox_seed, philox_offset, custom_mask_type,
    bias_requires_grad, scale, num_splits_key.value_or(0), window_size, shared_storage_dqdkdv);
#endif // USE_PPU
#endif // defined(USE_MEM_EFF_ATTENTION)
  TORCH_CHECK(false, "USE_MEM_EFF_ATTENTION was not enabled for build.")
  return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _scaled_dot_product_flash_attention_backward_cuda(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale){
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }

  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  Tensor grad_out_t = grad_out_.transpose(1,2);
  Tensor out_t = out.transpose(1,2);

  auto [grad_q, grad_k, grad_v] = at::_flash_attention_backward(
    grad_out_t,
    q_t,
    k_t,
    v_t,
    out_t,
    logsumexp,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_seqlen_batch_q,
    max_seqlen_batch_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    scale);

  grad_q = grad_q.transpose(1,2);
  grad_k = grad_k.transpose(1,2);
  grad_v = grad_v.transpose(1,2);

  return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v));
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> _scaled_dot_product_efficient_attention_backward_cuda(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    double dropout_p,
    std::array<bool, 4> grad_input_mask,
    bool causal,
    std::optional<double> scale) {

  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }
  constexpr int64_t MAX_BATCH_SIZE = (1LL << 16) - 1;
  int64_t batch_size = query.size(0);

  if (batch_size > MAX_BATCH_SIZE) {
    TORCH_CHECK(dropout_p == 0.0,
                "Efficient attention backward cannot handle dropout when "
                "the batch size exceeds (", MAX_BATCH_SIZE, ").");
  }
  auto grad_out_t = grad_out_.transpose(1, 2);
  auto query_t = query.transpose(1, 2);
  auto key_t = key.transpose(1, 2);
  auto value_t = value.transpose(1, 2);
  auto out_t = out.transpose(1, 2);

  auto process_chunk = [&](const Tensor& grad_out_chunk,
                          const Tensor& query_chunk,
                          const Tensor& key_chunk,
                          const Tensor& value_chunk,
                          const std::optional<Tensor>& attn_bias_chunk,
                          const Tensor& out_chunk,
                          const Tensor& logsumexp_chunk)
      -> std::tuple<Tensor, Tensor, Tensor, Tensor> {
  // This is needed because SaveVariable automatically converts
  // std::optional to undefined tensor
  std::optional<Tensor> kernel_bias;
  if (attn_bias_chunk.has_value() && attn_bias_chunk.value().defined()) {
    kernel_bias = attn_bias_chunk.value();
  }
  // Will add with signauter changes for dropout and bias
  // We are only handling Dense inputs, but this should be passed
  // from forward to backward
  int64_t max_seqlen_q = query_chunk.size(2);
  int64_t max_seqlen_k = key_chunk.size(2);

  sdp::CustomMaskType custom_mask_type = causal
    ? sdp::CustomMaskType::CausalFromTopLeft
    : sdp::CustomMaskType::NoCustomMask;
  auto [grad_q, grad_k, grad_v, grad_bias] =
      at::_efficient_attention_backward(
          grad_out_chunk,
          query_chunk,
          key_chunk,
          value_chunk,
          kernel_bias,
          out_chunk,
          std::nullopt,
          std::nullopt,
          max_seqlen_q,
          max_seqlen_k,
          logsumexp_chunk,
          dropout_p,
          philox_seed,
          philox_offset,
          static_cast<int64_t>(custom_mask_type),
          grad_input_mask[3],
          scale,
          std::nullopt);  // num_split_keys
  return std::make_tuple(
      grad_q.transpose(1, 2), grad_k.transpose(1, 2), grad_v.transpose(1, 2), std::move(grad_bias));
  };

  // process in chunks if batch size exceeds maximum
  if (batch_size > MAX_BATCH_SIZE) {
    Tensor final_grad_q, final_grad_k, final_grad_v, final_grad_bias;

    auto create_strided_output = [batch_size](const Tensor& tensor) -> Tensor {
      if (!tensor.defined()) {
        return Tensor{};
      }
      int dim = tensor.dim();
      std::vector<int64_t> sizes;
      sizes.reserve(dim);
      sizes.push_back(batch_size);
      for (int i = 1; i < dim; i++) {
        sizes.push_back(tensor.size(i));
      }
      return at::empty_strided(std::move(sizes), tensor.strides(), tensor.options());
    };

    if (grad_input_mask[0]) {
      final_grad_q = create_strided_output(query);
    }

    if (grad_input_mask[1]) {
      final_grad_k = create_strided_output(key);
    }

    if (grad_input_mask[2]) {
      final_grad_v = create_strided_output(value);
    }
    if (grad_input_mask[3] && attn_bias.defined()) {
      final_grad_bias = at::zeros_like(attn_bias);
    }

    for (int64_t start = 0; start < batch_size; start += MAX_BATCH_SIZE) {
      int64_t end = std::min(start + MAX_BATCH_SIZE, batch_size);

      Tensor grad_out_chunk = grad_out_t.slice(0, start, end);
      Tensor query_chunk = query_t.slice(0, start, end);
      Tensor key_chunk = key_t.slice(0, start, end);
      Tensor value_chunk = value_t.slice(0, start, end);
      Tensor attn_bias_chunk;
      if (attn_bias.defined()) {
        attn_bias_chunk = attn_bias.slice(0, start, end);
      } else {
        attn_bias_chunk.reset();
      }
      Tensor out_chunk = out_t.slice(0, start, end);
      Tensor logsumexp_chunk = logsumexp.numel() > 0 ? logsumexp.slice(0, start, end) : logsumexp;

      auto [chunk_grad_q, chunk_grad_k, chunk_grad_v, chunk_grad_bias] =
          process_chunk(grad_out_chunk, query_chunk, key_chunk, value_chunk,
                      attn_bias_chunk, out_chunk, logsumexp_chunk);

      if (grad_input_mask[0] && chunk_grad_q.defined()) {
        final_grad_q.slice(0, start, end).copy_(chunk_grad_q);
      }
      if (grad_input_mask[1] && chunk_grad_k.defined()) {
        final_grad_k.slice(0, start, end).copy_(chunk_grad_k);
      }
      if (grad_input_mask[2] && chunk_grad_v.defined()) {
        final_grad_v.slice(0, start, end).copy_(chunk_grad_v);
      }
      if (grad_input_mask[3] && chunk_grad_bias.defined()) {
        final_grad_bias.add_(chunk_grad_bias);
      }
    }

    return std::make_tuple(
        std::move(final_grad_q),
        std::move(final_grad_k),
        std::move(final_grad_v),
        std::move(final_grad_bias));
  }
  // when batch size is within allowed size, no chunking needed
  else {
    std::optional<Tensor> attn_bias_opt;
    if (attn_bias.defined()) {
      attn_bias_opt = attn_bias;
    }
    return process_chunk(grad_out_t, query_t, key_t, value_t, attn_bias_opt, out_t, logsumexp);
  }
}

std::tuple<Tensor, Tensor, Tensor> _scaled_dot_product_cudnn_attention_backward_cuda(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    const Tensor& attn_bias,
    const Tensor& cum_seq_q,
    const Tensor& cum_seq_k,
    const int64_t max_q,
    const int64_t max_k,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
        return at::_cudnn_attention_backward(
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            philox_seed,
            philox_offset,
            attn_bias,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            scale);
}

} // namespace at::native
