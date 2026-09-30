// PPU modification: lazy dlopen loader for the FA3 arbitrary-mask library
// (libflex_flash_attention.so), used by the flex_flash_attention SDPA
// backend.
//
// The library is deliberately NOT linked into libtorch at build time: it
// carries ~200 undefined libtorch symbols (c10::cuda::*, at::cuda::*) that
// resolve lazily against the RTLD_GLOBAL libtorch already in the process —
// the same contract as any torch C++ extension.  A build-time link would
// record the so in torch_cpu/torch_cuda's DT_NEEDED, and every downstream
// executable (test binaries, torch_shm_manager) would then fail to link on
// those unresolved symbols.  Instead, the call sites in attention.cpp and
// cuda/sdp_utils.cpp obtain the entry points here via dlopen/dlsym at
// first use; the wrappers mirror the contract in
// third_party/flex-flash-attention/flex_flash_attention/include/
// flex_flash_attention_sdpa.h.
//
// Library location: $FLEX_FLASH_ATTENTION_SO_PATH when set, otherwise
// libflex_flash_attention.so next to the loading libtorch so (cmake installs it
// into the same lib dir), falling back to the default search path.
//
// The dlsym names are the Itanium manglings of the contract declarations
// (verified via `nm -D libflex_flash_attention.so`).  If a signature changes,
// regenerate them from the so.
#pragma once

#include <dlfcn.h>

#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>

#include <c10/util/Exception.h>

// Forward-declare at::Tensor instead of including ATen headers: the torch
// side compiles with -DAT_PER_OPERATOR_HEADERS, which forbids umbrella
// includes like <ATen/ATen.h>; every caller already holds a Tensor type.
// cppcheck-suppress syntaxError ; CI C_CPPCHECK pass scans .h without --language=c++ (false positive here)
namespace at {
class Tensor;
}

// Top-level namespace on purpose: the call sites in attention.cpp and
// cuda/sdp_utils.cpp sit in different namespace scopes, and the original
// contract header also lived at top level (flex_flash_attention::*).
namespace flex_flash_attention_loader {

inline void* lib_handle() {
  static void* handle = [] {
    if (const char* env = std::getenv("FLEX_FLASH_ATTENTION_SO_PATH")) {
      void* h = dlopen(env, RTLD_NOW | RTLD_GLOBAL);
      const char* err = h ? nullptr : dlerror();
      TORCH_CHECK(
          h != nullptr,
          "FLEX_FLASH_ATTENTION_SO_PATH=",
          env,
          " could not be loaded: ",
          err ? err : "unknown error");
      return h;
    }
    // Candidate paths: next to the loading libtorch so (install layout),
    // then the default search path.
    std::string dir;
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&lib_handle), &info) &&
        info.dli_fname) {
      std::string path(info.dli_fname);
      auto pos = path.find_last_of('/');
      if (pos != std::string::npos) {
        dir = path.substr(0, pos + 1);
      }
    }
    auto try_open = [&]() -> void* {
      if (!dir.empty()) {
        if (void* h =
                dlopen((dir + "libflex_flash_attention.so").c_str(),
                       RTLD_NOW | RTLD_GLOBAL)) {
          return h;
        }
      }
      return dlopen("libflex_flash_attention.so", RTLD_NOW | RTLD_GLOBAL);
    };
    if (void* h = try_open()) {
      return h;
    }
    // Python loads libtorch_python.so with RTLD_LOCAL, so the libtorch
    // symbols this library needs (c10::cuda::*, at::cuda::*, the
    // torch::autograd::AutogradMeta vtable, ...) sit in a LOCAL scope and
    // RTLD_NOW resolution fails.  Re-opening libtorch_python.so with
    // RTLD_GLOBAL promotes its dependency closure (libtorch_cpu/cuda,
    // libc10*) into the global scope — the same effect a torch C++
    // extension gets through its DT_NEEDED on libtorch_python.so.  The so
    // is already loaded, so this only bumps its refcount and scope mode.
    if (!dir.empty()) {
      // Non-fatal: if this re-open fails, the resolve() caller still reports a
      // hard error, but its message blames libflex_flash_attention.so and hides
      // this root cause -- so warn here with the real dlerror().
      if (!dlopen((dir + "libtorch_python.so").c_str(),
                  RTLD_NOW | RTLD_GLOBAL)) {
        const char* err = dlerror();
        TORCH_WARN(
            "Failed to re-open ",
            dir,
            "libtorch_python.so with RTLD_GLOBAL: ",
            err ? err : "unknown error",
            "; the FLEX_FLASH_ATTENTION backend may be unavailable.");
      }
    }
    return try_open();
  }();
  return handle;
}

template <typename Fn>
inline Fn* resolve(const char* mangled) {
  void* h = lib_handle();
  const char* err = h ? nullptr : dlerror();
  TORCH_CHECK(
      h != nullptr,
      "libflex_flash_attention.so could not be loaded (needed by the "
      "FLEX_FLASH_ATTENTION SDPA backend): ",
      err ? err : "unknown error");
  void* sym = dlsym(h, mangled);
  err = sym ? nullptr : dlerror();
  TORCH_CHECK(
      sym != nullptr,
      "symbol ",
      mangled,
      " not found in libflex_flash_attention.so: ",
      err ? err : "unknown error");
  return reinterpret_cast<Fn*>(sym);
}

inline bool sdpa_available() {
  static auto* fn =
      resolve<bool()>("_ZN20flex_flash_attention14sdpa_availableEv");
  return fn();
}

inline bool sdpa_mask_decomposable(const at::Tensor& attn_mask) {
  static auto* fn = resolve<bool(const at::Tensor&)>(
      "_ZN20flex_flash_attention22sdpa_mask_decomposableERKN2at6TensorE");
  return fn(attn_mask);
}

// CUDA-graph contract (see flex_flash_attention_sdpa.h): capture-time verdict
// lookup (pure host, no device syncs) and the post-replay envelope check.
inline int sdpa_mask_verdict_cached(const at::Tensor& attn_mask) {
  static auto* fn = resolve<int(const at::Tensor&)>(
      "_ZN20flex_flash_attention24sdpa_mask_verdict_cachedERKN2at6TensorE");
  return fn(attn_mask);
}

inline void sdpa_graph_mask_check() {
  static auto* fn =
      resolve<void()>("_ZN20flex_flash_attention21sdpa_graph_mask_checkEv");
  fn();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_fwd(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    bool is_causal,
    double dropout_p,
    std::optional<double> scale) {
  static auto* fn = resolve<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
      const at::Tensor&,
      const at::Tensor&,
      const at::Tensor&,
      const std::optional<at::Tensor>&,
      bool,
      double,
      std::optional<double>)>(
      "_ZN20flex_flash_attention8sdpa_fwdERKN2at6TensorES3_S3_RKSt8optionalIS1_EbdS4_IdE");
  return fn(query, key, value, attn_mask, is_causal, dropout_p, scale);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_bwd(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    std::array<bool, 3> grad_input_mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& rng_state,
    bool is_causal,
    double dropout_p,
    std::optional<double> scale) {
  static auto* fn = resolve<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
      const at::Tensor&,
      const at::Tensor&,
      const at::Tensor&,
      const at::Tensor&,
      const std::optional<at::Tensor>&,
      std::array<bool, 3>,
      const at::Tensor&,
      const at::Tensor&,
      const at::Tensor&,
      bool,
      double,
      std::optional<double>)>(
      "_ZN20flex_flash_attention8sdpa_bwdERKN2at6TensorES3_S3_S3_RKSt8optionalIS1_ESt5arrayIbLm3EES3_S3_S3_bdS4_IdE");
  return fn(
      grad_out,
      query,
      key,
      value,
      attn_mask,
      grad_input_mask,
      out,
      logsumexp,
      rng_state,
      is_causal,
      dropout_p,
      scale);
}

} // namespace flex_flash_attention_loader
