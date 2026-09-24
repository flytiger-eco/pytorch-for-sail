# PPU modification (included from caffe2/CMakeLists.txt only under USE_PPU)
# Note [Flex Flash Attention SDPA backend integration]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Integrates the FA3 arbitrary-mask library (third_party/flex-flash-attention
# submodule) as a plain shared library behind the flex_flash_attention
# SDPA backend.  No wheel, no runtime op registration: the fork's CUDA
# dispatch of aten::_scaled_dot_product_flex_flash_attention(+_backward)
# calls flex_flash_attention::sdpa_fwd/bwd from
# third_party/flex-flash-attention/flex_flash_attention/include/flex_flash_attention_sdpa.h
# directly (see attention.cpp, guarded by USE_FLEX_FLASH_ATTENTION).
#
# The submodule ships in PPU-original form (hggc runtime APIs); setup.py
# converts it IN PLACE with third_party/cudafy-for-sail before this file is
# processed (run_cudafy_once, same treatment as third_party/flash-attention
# and third_party/cutlass).  The kernel-side cutlass3 headers come from the
# nested flex-flash-attention/csrc/actlize submodule (t-head/actlize v1.0.0),
# cudafied in place by setup.py as well.
#
# Build flow:
#   1. flex_flash_attention/build_lib.py (torch-import-free, nvcc/g++
#      direct) compiles libflex_flash_attention.so from the converted
#      submodule tree;
#   2. the so is installed next to libtorch and loaded at runtime via
#      dlopen/dlsym from aten/src/ATen/native/transformers/
#      flex_flash_attention_loader.h.  It is deliberately NOT linked into
#      torch_cpu/torch_cuda: it carries ~200 undefined libtorch symbols
#      (c10::cuda::*, at::cuda::*) that resolve lazily against the
#      RTLD_GLOBAL libtorch already in the process (same contract as any
#      torch C++ extension); a build-time link would record the so in
#      DT_NEEDED and make every downstream executable (test binaries,
#      torch_shm_manager) fail to link on those unresolved symbols.
#
# If the submodule is absent (not fetched) the backend is compiled out
# (NOT_IMPLEMENTED stubs remain) and nothing else is affected.
#
# Porting to another torch version: copy this file, the single include(...)
# line in caffe2/CMakeLists.txt, the USE_FLEX_FLASH_ATTENTION hook in setup.py,
# and the USE_FLEX_FLASH_ATTENTION hunks in native_functions.yaml /
# attention.cpp / sdp_utils.cpp.

option(USE_FLEX_FLASH_ATTENTION
       "Link the arbitrary-mask flash attention library from third_party/flex-flash-attention into the SDPA backend"
       ON)

if(NOT USE_FLEX_FLASH_ATTENTION)
  return()
endif()

if(NOT USE_CUDA)
  message(STATUS "USE_FLEX_FLASH_ATTENTION requires USE_CUDA; skipping")
  return()
endif()

# Library source root.  Default: third_party/flex-flash-attention submodule.
# Point this at any cudafied checkout of the FA3 arbitrary-mask repo
# (e.g. a standalone clone):
#   cmake ... -DFLEX_FLASH_ATTENTION_ROOT=/path/to/checkout
if(DEFINED ENV{FLEX_FLASH_ATTENTION_ROOT})
  set(_fa_default_root "$ENV{FLEX_FLASH_ATTENTION_ROOT}")
else()
  set(_fa_default_root "${CMAKE_SOURCE_DIR}/third_party/flex-flash-attention")
endif()
set(FLEX_FLASH_ATTENTION_ROOT "${_fa_default_root}"
    CACHE PATH "Root of the FA3 arbitrary-mask library repo")
# The module ships either as a subdir of the fork (nested layout:
# <root>/flex_flash_attention/build_lib.py) or promoted to the repo root
# (flattened layout: <root>/build_lib.py).  Detect which and use it below.
if(EXISTS "${FLEX_FLASH_ATTENTION_ROOT}/build_lib.py")
  set(_fa_module_dir "${FLEX_FLASH_ATTENTION_ROOT}")
else()
  set(_fa_module_dir "${FLEX_FLASH_ATTENTION_ROOT}/flex_flash_attention")
endif()
set(FLEX_FLASH_ATTENTION_BUILD_SCRIPT
    "${_fa_module_dir}/build_lib.py")
set(FLEX_FLASH_ATTENTION_INCLUDE
    "${_fa_module_dir}/include")

if(NOT EXISTS "${FLEX_FLASH_ATTENTION_BUILD_SCRIPT}")
  message(STATUS
          "flex_flash_attention library not found at ${FLEX_FLASH_ATTENTION_ROOT} "
          "(missing ${FLEX_FLASH_ATTENTION_BUILD_SCRIPT}); populate the "
          "third_party/flex-flash-attention submodule or set "
          "-DFLEX_FLASH_ATTENTION_ROOT=<repo checkout>; skipping SDPA integration")
  return()
endif()

# setup.py runs the cudafy conversion before cmake configure; warn (not
# fail) when its stamp is missing, e.g. on a pure-cmake reconfigure after
# a `git submodule update` reset the tree.
if(NOT EXISTS
   "${FLEX_FLASH_ATTENTION_ROOT}/.cudafy-for-sail/flex-flash-attention-2.8.2.stamp")
  message(WARNING
          "flex_flash_attention sources at ${FLEX_FLASH_ATTENTION_ROOT} do not look "
          "cudafied (no .cudafy-for-sail stamp); build torch via setup.py "
          "or run third_party/cudafy-for-sail/cudafy.py flex-flash-attention "
          "--version=2.8.2 on the tree, otherwise the kernel compile will "
          "fail on hggc sources.")
endif()

set(FLEX_FLASH_ATTENTION_SO "${CMAKE_BINARY_DIR}/lib/libflex_flash_attention.so")
set(FLEX_FLASH_ATTENTION_OBJS "${CMAKE_BINARY_DIR}/flex_flash_attention_objs")

# Torch headers for the library build: reuse the exact include set
# torch_cpu compiles with (aten/src + codegen output dirs).  Note: do NOT
# use Caffe2_CPU_INCLUDE here — ATen_CPU_INCLUDE is appended into it early
# in caffe2/CMakeLists.txt before it is populated.
set(_fa_include_args "")
foreach(_inc ${ATen_CPU_INCLUDE})
  list(APPEND _fa_include_args "--include" "${_inc}")
endforeach()
# Repo root: c10/ and torch/ headers live there (aten/src alone only
# covers ATen/...).  torch/nn (torch::nn::functional) headers used by the
# library API TU come from torch/csrc/api/include.
list(APPEND _fa_include_args "--include" "${TORCH_ROOT}")
list(APPEND _fa_include_args "--include" "${TORCH_SRC_DIR}/csrc/api/include")
# Generated headers (cmake_macros.h under torch/headeronly/macros/, codegen
# output) live in the build tree, not the source tree.
list(APPEND _fa_include_args "--include" "${CMAKE_BINARY_DIR}")
# CUDA runtime headers (cuda_runtime_api.h) for the host-side API TU;
# ATen_CPU_INCLUDE only covers CPU headers.
if(CUDA_TOOLKIT_INCLUDE)
  list(APPEND _fa_include_args "--include" "${CUDA_TOOLKIT_INCLUDE}")
endif()
# Shell-quoted copy for the bash -c wrapper below.
set(_fa_quoted_includes "")
foreach(_arg ${_fa_include_args})
  string(APPEND _fa_quoted_includes " '${_arg}'")
endforeach()

# Rebuild the so when library sources change.  Globbed at configure time:
# add/remove files in the library then re-run cmake configure.
file(GLOB_RECURSE _fa_src_files
     "${_fa_module_dir}/csrc/*"
     "${_fa_module_dir}/instantiations/*"
     "${_fa_module_dir}/include/*")

# PPU_SDK must NOT reach build_lib.py: it prepends
# $PPU_SDK/targets/x86_64-linux/include (hggc_math_forward_declares.h) to
# the kernel include path, whose __device__ math declarations collide with
# GCC's constexpr overloads in <cmath> pulled in through torch headers.
# The PPU toolchain finds its own runtime headers without it.  (env -u is
# wrapped in bash because `cmake -E env -u` needs cmake >= 3.22.)
add_custom_command(
    OUTPUT "${FLEX_FLASH_ATTENTION_SO}"
    COMMAND bash -c "env -u PPU_SDK '${Python_EXECUTABLE}' '${FLEX_FLASH_ATTENTION_BUILD_SCRIPT}' --out '${FLEX_FLASH_ATTENTION_SO}' --obj-dir '${FLEX_FLASH_ATTENTION_OBJS}' --nvcc '${CMAKE_CUDA_COMPILER}' --cxx '${CMAKE_CXX_COMPILER}' ${_fa_quoted_includes} --define '_GLIBCXX_USE_CXX11_ABI=1'"
    # build_lib.py compiles against headers and never links libtorch; we
    # only need the generated headers from both ATen (Register/DispatchKey)
    # and torch (variable_factories.h etc.) to exist before our TU runs.
    DEPENDS ATEN_CPU_FILES_GEN_TARGET ATEN_CUDA_FILES_GEN_TARGET
            generate-torch-sources
            "${FLEX_FLASH_ATTENTION_BUILD_SCRIPT}" ${_fa_src_files}
    WORKING_DIRECTORY "${FLEX_FLASH_ATTENTION_ROOT}"
    COMMENT "Building libflex_flash_attention.so (arbitrary-mask SDPA library)")

add_custom_target(flex_flash_attention_lib ALL DEPENDS "${FLEX_FLASH_ATTENTION_SO}")

# No target_link_libraries here on purpose: see the build-flow note at
# the top.  The call sites (attention.cpp, cuda/sdp_utils.cpp) obtain the
# entry points via flex_flash_attention_loader.h at first use.  Keep the macro
# and the build ordering (the so must exist by the time torch is complete
# and gets packaged).
add_dependencies(torch_cuda flex_flash_attention_lib)
target_compile_definitions(torch_cuda PRIVATE USE_FLEX_FLASH_ATTENTION)
target_compile_definitions(torch_cpu PRIVATE USE_FLEX_FLASH_ATTENTION)

install(FILES "${FLEX_FLASH_ATTENTION_SO}" DESTINATION "${TORCH_INSTALL_LIB_DIR}")

message(STATUS
        "USE_FLEX_FLASH_ATTENTION: building ${FLEX_FLASH_ATTENTION_SO} "
        "(dlopen'd at runtime, not linked into libtorch)")

