// PPU elementwise kernel registration filter.
//
// Every kernel in torch_cuda is registered with the runtime by static
// initializers through __cudaRegisterFunction, and each registered kernel
// lengthens the runtime's per-launch lookup. The PPU kernels add tens of
// thousands of instantiations, so torch_cuda is linked with
// --wrap=__cudaRegisterFunction and they are dropped here unless every visible
// device is a Zhenwu M890 and PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT is "1".
// elementwise_ppu_enabled() follows the same decision, so a dropped kernel is
// never launched.

#include <cstdlib>
#include <cstring>

#include "utils/env.cuh"

namespace at::native {

bool elementwise_ppu_kernels_registered() {
  static const bool registered = []() {
    const char* value =
        std::getenv("PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT");
    if (value == nullptr || std::strcmp(value, "1") != 0) {
      return false;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      return false;
    }

    for (int device = 0; device < device_count; ++device) {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, device) != cudaSuccess ||
          prop.major != 8 || prop.minor != 9) {
        return false;
      }
    }
    return true;
  }();
  return registered;
}

} // namespace at::native

namespace {

// Mangled name prefix of everything declared in
// at::native::ppu_elementwise_opt.
constexpr char kPpuKernelPrefix[] = "_ZN2at6native19ppu_elementwise_opt";

} // namespace

extern "C" {

void __real___cudaRegisterFunction(
    void** fat_cubin_handle,
    const char* host_fun,
    char* device_fun,
    const char* device_name,
    int thread_limit,
    uint3* tid,
    uint3* bid,
    dim3* block_dim,
    dim3* grid_dim,
    int* warp_size);

void __wrap___cudaRegisterFunction(
    void** fat_cubin_handle,
    const char* host_fun,
    char* device_fun,
    const char* device_name,
    int thread_limit,
    uint3* tid,
    uint3* bid,
    dim3* block_dim,
    dim3* grid_dim,
    int* warp_size) {
  if (!at::native::elementwise_ppu_kernels_registered() &&
      std::strncmp(
          device_name, kPpuKernelPrefix, sizeof(kPpuKernelPrefix) - 1) == 0) {
    return;
  }
  __real___cudaRegisterFunction(
      fat_cubin_handle,
      host_fun,
      device_fun,
      device_name,
      thread_limit,
      tid,
      bid,
      block_dim,
      grid_dim,
      warp_size);
}

} // extern "C"
