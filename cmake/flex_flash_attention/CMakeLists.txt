# Isolate PyTorch's global CUDA architectures, half macros, and optimization flags in this directory.
# Read flags from the submodule build_lib.py; CMake manages compilation, depfiles, caching, and parallelism.
set(_fa_metadata_script "${TORCH_ROOT}/tools/build/ffa_build_metadata.py")
execute_process(
  COMMAND "${Python_EXECUTABLE}" "${_fa_metadata_script}" "${_fa_module_dir}"
  RESULT_VARIABLE _fa_metadata_status
  OUTPUT_VARIABLE _fa_metadata
  ERROR_VARIABLE _fa_metadata_error)
if(NOT _fa_metadata_status EQUAL 0)
  message(FATAL_ERROR "Cannot read FFA build metadata: ${_fa_metadata_error}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_fa_metadata_script}" "${_fa_module_dir}/build_lib.py"
  "${_fa_module_dir}/sources.py")

function(_fa_read_list key output)
  string(JSON _length LENGTH "${_fa_metadata}" "${key}")
  set(_items "")
  if(_length GREATER 0)
    math(EXPR _last "${_length} - 1")
    foreach(_index RANGE 0 ${_last})
      string(JSON _item GET "${_fa_metadata}" "${key}" ${_index})
      list(APPEND _items "${_item}")
    endforeach()
  endif()
  set(${output} "${_items}" PARENT_SCOPE)
endfunction()

_fa_read_list(sources _fa_sources)
_fa_read_list(includes _fa_kernel_includes)
_fa_read_list(cuda_flags _fa_cuda_flags)
_fa_read_list(cxx_flags _fa_cxx_flags)

# Limit these settings to this directory; preserve torch_cuda architectures and language standards.
foreach(_language CXX CUDA)
  set(CMAKE_${_language}_FLAGS "")
  foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_${_language}_FLAGS_${_config} "")
  endforeach()
endforeach()
set(CMAKE_CUDA_ARCHITECTURES OFF)
# Match the original standalone library: link with CXX only, without adding libtorch/CUDA dependencies.
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "")
add_library(flex_flash_attention_lib SHARED ${_fa_sources})
set_target_properties(flex_flash_attention_lib PROPERTIES
  OUTPUT_NAME flex_flash_attention
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES OFF CUDA_RUNTIME_LIBRARY None
  POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CXX)
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
target_include_directories(flex_flash_attention_lib PRIVATE
  ${_fa_kernel_includes} ${ATen_CPU_INCLUDE} "${TORCH_ROOT}"
  "${TORCH_SRC_DIR}/csrc/api/include" "${CMAKE_BINARY_DIR}"
  ${CUDA_TOOLKIT_INCLUDE})
# Reuse the parent CXX/CUDA launchers, including USE_CCACHE, without creating a separate process pool.
set_property(TARGET flex_flash_attention_lib PROPERTY CUDA_COMPILER_LAUNCHER
  "${CMAKE_COMMAND};-E;env;HGGC_ENABLE_COMPRESS=1;${CMAKE_CUDA_COMPILER_LAUNCHER}")
add_dependencies(flex_flash_attention_lib
  ATEN_CPU_FILES_GEN_TARGET ATEN_CUDA_FILES_GEN_TARGET generate-torch-sources)
