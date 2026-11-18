# Read the standalone command contract from build_lib.py, without inheriting caffe2's directory flags.
# These settings are local to this directory; torch_cuda's architectures, ABI and flags are unchanged.
torch_ffa_build_metadata("${_fa_module_dir}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_fa_module_dir}/build_lib.py" "${_fa_module_dir}/sources.py")

# Clearing language flags alone does not clear add_definitions/include_directories/add_compile_options.
# The standalone oracle did not receive these directory properties (including system include prefixes).
foreach(_property COMPILE_DEFINITIONS INCLUDE_DIRECTORIES COMPILE_OPTIONS LINK_OPTIONS LINK_DIRECTORIES)
  set_property(DIRECTORY PROPERTY ${_property} "")
endforeach()
foreach(_language CXX CUDA)
  set(CMAKE_${_language}_FLAGS "")
  unset(CMAKE_${_language}_STANDARD)
  unset(CMAKE_${_language}_STANDARD_REQUIRED)
  unset(CMAKE_${_language}_EXTENSIONS)
  foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_${_language}_FLAGS_${_config} "")
    set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS_${_config} "")
  endforeach()
endforeach()
# Do not add a diagnostic-color flag or another -std option to the parsed oracle flags.
unset(CMAKE_COLOR_DIAGNOSTICS)
set(CMAKE_CUDA_ARCHITECTURES OFF)
# The standalone link was only CXX -shared -Wl,-soname,<name> <objects> -o <library>.
# Preserve Ninja's linker depfile bookkeeping, but do not inherit torch's link flags/RPATH/lwyu.
set(CMAKE_SHARED_LINKER_FLAGS "")
foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_SHARED_LINKER_FLAGS_${_config} "")
endforeach()
set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "")
add_library(flex_flash_attention_lib SHARED ${_fa_sources})
set_target_properties(flex_flash_attention_lib PROPERTIES
  OUTPUT_NAME flex_flash_attention DEFINE_SYMBOL ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  CUDA_ARCHITECTURES OFF CUDA_RUNTIME_LIBRARY None
  POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CXX
  LINK_WHAT_YOU_USE FALSE SKIP_BUILD_RPATH TRUE
  INSTALL_RPATH "" INSTALL_RPATH_USE_LINK_PATH FALSE)
foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(flex_flash_attention_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${_config} "${CMAKE_BINARY_DIR}/lib")
endforeach()
# Keep repeated -mllvm/-gencode option pairs intact during CMake deduplication.
set(_fa_pair "")
foreach(_flag IN LISTS _fa_cuda_flags)
  if(_fa_pair)
    set(_option "SHELL:${_fa_pair} ${_flag}")
    set(_fa_pair "")
  elseif(_flag STREQUAL "-mllvm" OR _flag STREQUAL "-gencode")
    set(_fa_pair "${_flag}")
    continue()
  else()
    set(_option "${_flag}")
  endif()
  target_compile_options(flex_flash_attention_lib PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:${_option}>")
endforeach()
if(_fa_pair)
  message(FATAL_ERROR "Incomplete FFA compiler option: ${_fa_pair}")
endif()
foreach(_flag IN LISTS _fa_cxx_flags)
  target_compile_options(flex_flash_attention_lib PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${_flag}>")
endforeach()
if(DEFINED GLIBCXX_USE_CXX11_ABI)
  target_compile_definitions(flex_flash_attention_lib PRIVATE
    "_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
else()
  target_compile_definitions(flex_flash_attention_lib PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
endif()
# Explicit oracle include list, in the original order, without an inherited prefix.
target_include_directories(flex_flash_attention_lib PRIVATE
  ${_fa_kernel_includes} ${ATen_CPU_INCLUDE} "${TORCH_ROOT}"
  "${TORCH_SRC_DIR}/csrc/api/include" "${CMAKE_BINARY_DIR}"
  ${CUDA_TOOLKIT_INCLUDE})
# Reuse the parent CXX/CUDA launchers, including USE_CCACHE, without creating a separate process pool.
set_property(TARGET flex_flash_attention_lib PROPERTY CUDA_COMPILER_LAUNCHER
  "${CMAKE_COMMAND};-E;env;HGGC_ENABLE_COMPRESS=1;${CMAKE_CUDA_COMPILER_LAUNCHER}")
add_dependencies(flex_flash_attention_lib
  ATEN_CPU_FILES_GEN_TARGET ATEN_CUDA_FILES_GEN_TARGET generate-torch-sources)
