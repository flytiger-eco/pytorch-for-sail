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
#   1. a normal CMake SHARED target compiles each TU from the submodule
#      manifest, reusing launchers, depfiles, and parallel scheduling;
#      build_lib.py only supplies the manifest and flags at configure time;
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

# 只解析子模块的声明式清单子集，不执行 Python；结构变化必须显式报错。
function(_fa_tokens text output)
  set(_code "")
  set(_strings "")
  set(_layout "")
  set(_line_start TRUE)
  set(_indent 0)
  set(_depth 0)
  while(NOT "${text}" STREQUAL "")
    # 记录逻辑语句缩进；括号内换行和 docstring 不改变控制流。
    if(_line_start AND NOT text MATCHES "^([ \t\r\n#]|\"\"\"|''')")
      if(_depth EQUAL 0)
        list(APPEND _layout "${_indent}")
      endif()
      set(_line_start FALSE)
    endif()
    if(text MATCHES "^([ \t\r\n]+|#[^\n]*)")
      set(_space "${CMAKE_MATCH_0}")
      string(LENGTH "${_space}" _length)
      if(_space MATCHES "\n")
        string(REGEX REPLACE "^.*\n" "" _last_line "${_space}")
        if(_last_line MATCHES "\t")
          message(FATAL_ERROR "Unsupported FFA manifest tab indentation")
        endif()
        string(LENGTH "${_last_line}" _indent)
        set(_line_start TRUE)
      elseif(_line_start AND _space MATCHES "^[ ]+$")
        math(EXPR _indent "${_indent} + ${_length}")
      elseif(_line_start AND _space MATCHES "\t")
        message(FATAL_ERROR "Unsupported FFA manifest tab indentation")
      endif()
    elseif(text MATCHES "^(\"\"\"|''')")
      set(_quote "${CMAKE_MATCH_1}")
      if(NOT _line_start OR NOT _depth EQUAL 0)
        message(FATAL_ERROR "Unsupported FFA triple-quoted metadata value")
      endif()
      string(SUBSTRING "${text}" 3 -1 _tail)
      string(FIND "${_tail}" "${_quote}" _end)
      if(_end EQUAL -1)
        message(FATAL_ERROR "Unterminated FFA manifest docstring")
      endif()
      math(EXPR _length "${_end} + 6")
    elseif(text MATCHES "^(f?)(\"[^\"\n]*\"|'[^'\n]*')")
      set(_formatted "${CMAKE_MATCH_1}")
      set(_literal "${CMAKE_MATCH_2}")
      string(LENGTH "${CMAKE_MATCH_0}" _length)
      string(LENGTH "${_literal}" _size)
      math(EXPR _size "${_size} - 2")
      string(SUBSTRING "${_literal}" 1 ${_size} _literal)
      if(_literal MATCHES ";" OR _literal MATCHES "\\\\"
          OR _literal MATCHES "\\[" OR _literal MATCHES "\\]")
        message(FATAL_ERROR "Unsupported FFA manifest literal (escape, semicolon, or bracket)")
      endif()
      list(LENGTH _strings _index)
      # 空字符串也必须保留一个列表槽位。
      list(APPEND _strings "literal:${_literal}")
      if(_formatted)
        string(APPEND _code "F${_index}")
      else()
        string(APPEND _code "S${_index}")
      endif()
    else()
      string(SUBSTRING "${text}" 0 1 _character)
      string(APPEND _code "${_character}")
      if(_character STREQUAL "(" OR _character STREQUAL "[")
        math(EXPR _depth "${_depth} + 1")
      elseif(_character STREQUAL ")" OR _character STREQUAL "]")
        math(EXPR _depth "${_depth} - 1")
      endif()
      set(_length 1)
    endif()
    string(SUBSTRING "${text}" ${_length} -1 text)
  endwhile()
  set(${output} "${_code}" PARENT_SCOPE)
  set(${output}_strings "${_strings}" PARENT_SCOPE)
  set(${output}_layout "${_layout}" PARENT_SCOPE)
endfunction()

function(_fa_literal strings token output)
  string(REGEX REPLACE "^[SF]" "" _index "${token}")
  list(GET ${strings} ${_index} _value)
  string(SUBSTRING "${_value}" 8 -1 _value)
  set(${output} "${_value}" PARENT_SCOPE)
endfunction()

function(_fa_block text name output)
  string(REGEX MATCHALL "(^|\n)def ${name}\\(" _definitions "${text}")
  list(LENGTH _definitions _count)
  if(NOT _count EQUAL 1)
    message(FATAL_ERROR "Missing or duplicate FFA manifest function: ${name}")
  endif()
  string(REGEX MATCH "(^|\n)def ${name}\\([^\n]*(\n([ \t]+[^\n]*)?)*" _block "${text}")
  if(NOT _block)
    message(FATAL_ERROR "Missing FFA manifest function: ${name}")
  endif()
  set(${output} "${_block}" PARENT_SCOPE)
endfunction()

function(_fa_list text name output)
  string(REGEX MATCH "(^|\n)${name}[ \t]*=[ \t]*\\[[^]]*\\]" _block "${text}")
  _fa_tokens("${_block}" _tokens)
  if(NOT _tokens MATCHES "^${name}=\\[(S[0-9]+(,S[0-9]+)*,?)?\\]$")
    message(FATAL_ERROR "FFA ${name} must be a literal string list")
  endif()
  string(REGEX MATCHALL "S[0-9]+" _items "${_tokens}")
  set(_values "")
  foreach(_item IN LISTS _items)
    _fa_literal(_tokens_strings "${_item}" _value)
    if(_value STREQUAL "")
      message(FATAL_ERROR "Unsupported FFA empty compiler option: ${name}")
    endif()
    list(APPEND _values "${_value}")
  endforeach()
  set(${output} "${_values}" PARENT_SCOPE)
endfunction()

function(_fa_source_template strings token kind dimensions output)
  _fa_literal(${strings} "${token}" _template)
  if((kind STREQUAL "" AND _template MATCHES "\\{kind\\}")
      OR (dimensions STREQUAL "once" AND _template MATCHES "\\{h\\}"))
    message(FATAL_ERROR "Unsupported FFA source interpolation context: ${_template}")
  endif()
  string(REPLACE "{_DIR}" "${_fa_instantiations}" _template "${_template}")
  string(REPLACE "{kind}" "${kind}" _template "${_template}")
  set(_values "")
  foreach(_hdim IN LISTS dimensions)
    string(REPLACE "{h}" "${_hdim}" _value "${_template}")
    if(_value MATCHES "[{}]")
      message(FATAL_ERROR "Unsupported FFA source interpolation: ${_value}")
    endif()
    list(APPEND _values "${_value}")
  endforeach()
  set(${output} "${_values}" PARENT_SCOPE)
endfunction()

function(_fa_source_group text name kind output)
  _fa_block("${text}" "${name}" _block)
  _fa_tokens("${_block}" _tokens)
  if(NOT _tokens_layout MATCHES "^0(;4)+$")
    message(FATAL_ERROR "Unsupported FFA source group indentation: ${name}")
  endif()
  if(name STREQUAL "_sources")
    set(_signature "kind,disabled_hdims")
  else()
    set(_signature "disabled_hdims")
  endif()
  if(NOT _tokens MATCHES "^def${name}\\(${_signature}\\):out=\\[(F[0-9]+)\\]")
    message(FATAL_ERROR "Unsupported FFA source group: ${name}")
  endif()
  set(_prefix "${CMAKE_MATCH_0}")
  set(_token "${CMAKE_MATCH_1}")
  _fa_source_template(_tokens_strings "${_token}" "${kind}" "once" _values)
  string(LENGTH "${_prefix}" _length)
  string(SUBSTRING "${_tokens}" ${_length} -1 _rest)
  while(NOT _rest STREQUAL "returnout")
    if(_rest MATCHES "^out\\.append\\((F[0-9]+)\\)")
      set(_dimensions once)
    elseif(_rest MATCHES "^out\\+=\\[(F[0-9]+)forhinHEAD_DIMENSIONSifhnotindisabled_hdims\\]")
      set(_dimensions "${_fa_head_dimensions}")
    else()
      message(FATAL_ERROR "Unsupported FFA ${name} statement: ${_rest}")
    endif()
    set(_prefix "${CMAKE_MATCH_0}")
    set(_token "${CMAKE_MATCH_1}")
    _fa_source_template(_tokens_strings "${_token}" "${kind}" "${_dimensions}" _items)
    list(APPEND _values ${_items})
    string(LENGTH "${_prefix}" _length)
    string(SUBSTRING "${_rest}" ${_length} -1 _rest)
  endwhile()
  set(${output} "${_values}" PARENT_SCOPE)
endfunction()

function(torch_ffa_build_metadata module_dir)
  foreach(_path "${module_dir}" "$ENV{ACTLIZE_DIR}")
    if(_path MATCHES ";" OR _path MATCHES "\\[" OR _path MATCHES "\\]")
      message(FATAL_ERROR "Unsupported FFA metadata path: ${_path}")
    endif()
  endforeach()
  file(REAL_PATH "${module_dir}" _module)
  file(READ "${_module}/build_lib.py" _build)
  file(READ "${_module}/sources.py" _manifest)
  # 检查 import 阶段的骨架；防止常量在稍后被重赋值或动态修改。
  set(_build_globals "${_build}")
  foreach(_name kernel_include_dirs sources header_stamp compile_one main)
    _fa_block("${_build}" "${_name}" _block)
    string(REPLACE "${_block}" "\n" _build_globals "${_build_globals}")
  endforeach()
  foreach(_name NVCC_FLAGS GENCODE CXX_FLAGS)
    string(REGEX MATCH "(^|\n)${_name}[ \t]*=[ \t]*\\[[^]]*\\]" _block "${_build_globals}")
    string(REPLACE "${_block}" "\n" _build_globals "${_build_globals}")
  endforeach()
  foreach(_name THIS_DIR REPO_DIR)
    string(REGEX MATCH "(^|\n)${_name}[ \t]*=[^\n]*" _block "${_build_globals}")
    string(REPLACE "${_block}" "\n" _build_globals "${_build_globals}")
  endforeach()
  _fa_tokens("${_build_globals}" _tokens)
  if(NOT _tokens STREQUAL "importargparseimportosimportsubprocessimportsysfromconcurrent.futuresimportProcessPoolExecutorfrompathlibimportPathHOPPER_DIR=REPO_DIR/S0importimportlib.util_spec=importlib.util.spec_from_file_location(S1,THIS_DIR/S2)_sources_mod=importlib.util.module_from_spec(_spec)_spec.loader.exec_module(_sources_mod)flex_flash_sources=_sources_mod.flex_flash_sources_HDR_STAMP=Noneif__name__==S3:main()")
    message(FATAL_ERROR "Unsupported FFA build metadata globals")
  endif()
  _fa_literal(_tokens_strings S2 _source_module)
  _fa_literal(_tokens_strings S3 _main)
  if(NOT _source_module STREQUAL "sources.py" OR NOT _main STREQUAL "__main__")
    message(FATAL_ERROR "Unsupported FFA source module import")
  endif()
  # 清单的全局声明只有目录和 head-dim；不接受额外的顶层变更逻辑。
  set(_globals "${_manifest}")
  foreach(_name _sources _sources_f32 flex_flash_sources)
    _fa_block("${_manifest}" "${_name}" _block)
    string(REPLACE "${_block}" "\n" _globals "${_globals}")
  endforeach()
  _fa_tokens("${_globals}" _tokens)
  if(NOT _tokens_layout MATCHES "^0(;0)*$")
    message(FATAL_ERROR "Unsupported FFA source manifest indentation")
  endif()
  if(NOT _tokens MATCHES "^_DIR=S0HEAD_DIMENSIONS=\\[([0-9]+(,[0-9]+)*,?)\\]$")
    message(FATAL_ERROR "Unsupported FFA source manifest globals")
  endif()
  string(REPLACE "," ";" _fa_head_dimensions "${CMAKE_MATCH_1}")
  list(FILTER _fa_head_dimensions EXCLUDE REGEX "^$")
  _fa_literal(_tokens_strings S0 _fa_instantiations)
  # 支持 flattened 以及以 THIS_DIR.parent 定位 repo root 的 nested 布局。
  string(REGEX MATCH "(^|\n)THIS_DIR[ \t]*=[^\n]*" _this_dir "${_build}")
  _fa_tokens("${_this_dir}" _this_dir)
  if(NOT _this_dir STREQUAL "THIS_DIR=Path(__file__).resolve().parent")
    message(FATAL_ERROR "Unsupported FFA THIS_DIR expression")
  endif()
  string(REGEX MATCH "(^|\n)REPO_DIR[ \t]*=[^\n]*" _repo_dir "${_build}")
  _fa_tokens("${_repo_dir}" _repo_dir)
  if(NOT _repo_dir MATCHES "^REPO_DIR=THIS_DIR(\\.parent)*$")
    message(FATAL_ERROR "Unsupported FFA REPO_DIR expression")
  endif()
  set(_repo "${_module}")
  string(REGEX MATCHALL "\\.parent" _parents "${_repo_dir}")
  foreach(_parent IN LISTS _parents)
    get_filename_component(_repo "${_repo}" DIRECTORY)
  endforeach()

  _fa_block("${_manifest}" flex_flash_sources _block)
  _fa_tokens("${_block}" _tokens)
  if(NOT _tokens_layout STREQUAL "0;4;4;4;8;4;4")
    message(FATAL_ERROR "Unsupported FFA flex_flash_sources indentation")
  endif()
  if(NOT _tokens MATCHES "^defflex_flash_sources\\(disabled_hdims=\\(\\),disable_backward=False,disable_sm8x=False\\):srcs=\\[((S[0-9]+)(,S[0-9]+)*,?)\\]srcs\\+=_sources\\((S[0-9]+),disabled_hdims\\)ifnotdisable_backwardandnotdisable_sm8x:srcs\\+=_sources\\((S[0-9]+),disabled_hdims\\)srcs\\+=_sources_f32\\(disabled_hdims\\)returnsrcs$")
    message(FATAL_ERROR "Unsupported FFA flex_flash_sources structure")
  endif()
  set(_initial "${CMAKE_MATCH_1}")
  set(_forward "${CMAKE_MATCH_4}")
  set(_backward "${CMAKE_MATCH_5}")
  _fa_literal(_tokens_strings "${_forward}" _forward)
  _fa_literal(_tokens_strings "${_backward}" _backward)
  set(_sources "")
  string(REGEX MATCHALL "S[0-9]+" _items "${_initial}")
  foreach(_item IN LISTS _items)
    _fa_literal(_tokens_strings "${_item}" _value)
    list(APPEND _sources "${_value}")
  endforeach()
  foreach(_kind "${_forward}" "${_backward}")
    _fa_source_group("${_manifest}" _sources "${_kind}" _items)
    list(APPEND _sources ${_items})
  endforeach()
  _fa_source_group("${_manifest}" _sources_f32 "" _items)
  list(APPEND _sources ${_items})

  _fa_block("${_build}" sources _block)
  _fa_tokens("${_block}" _tokens)
  # 这里只检查结构，路径和排除前缀仍从子模块字面量读取。
  if(NOT _tokens MATCHES "^defsources\\(\\):srcs=\\[sforsinflex_flash_sources\\(\\)ifnots\\.startswith\\(S0\\)\\]srcs=\\[REPO_DIR/sforsinsrcs\\]srcs\\.append\\(THIS_DIR((/S[0-9]+)+)\\)returnsrcs$"
      OR NOT _tokens_layout STREQUAL "0;4;4;4;4")
    message(FATAL_ERROR "Unsupported FFA library sources structure")
  endif()
  set(_glue_tokens "${CMAKE_MATCH_1}")
  _fa_literal(_tokens_strings S0 _excluded)
  set(_glue "${_module}")
  string(REGEX MATCHALL "S[0-9]+" _items "${_glue_tokens}")
  foreach(_item IN LISTS _items)
    _fa_literal(_tokens_strings "${_item}" _part)
    cmake_path(APPEND _glue "${_part}")
  endforeach()
  set(_absolute_sources "")
  foreach(_source IN LISTS _sources)
    string(FIND "${_source}" "${_excluded}" _position)
    if(NOT _position EQUAL 0)
      set(_path "${_repo}")
      cmake_path(APPEND _path "${_source}")
      file(REAL_PATH "${_path}" _absolute)
      list(APPEND _absolute_sources "${_absolute}")
    endif()
  endforeach()
  file(REAL_PATH "${_glue}" _absolute)
  list(APPEND _absolute_sources "${_absolute}")
  foreach(_source IN LISTS _absolute_sources)
    if(NOT EXISTS "${_source}" OR IS_DIRECTORY "${_source}")
      message(FATAL_ERROR "FFA source manifest contains nonexistent files: ${_source}")
    endif()
  endforeach()

  _fa_block("${_build}" kernel_include_dirs _block)
  _fa_tokens("${_block}" _tokens)
  # 固定控制流门禁防止上游更改路径优先级后被静默忽略。
  _fa_tokens([=[
def kernel_include_dirs():
    dirs = [str(THIS_DIR / ""), str(REPO_DIR)]
    candidates = []
    env_actlize = os.environ.get("", "")
    if env_actlize:
        candidates.append(Path(env_actlize))
    candidates.append(REPO_DIR / "" / "")
    for cand in candidates:
        if (cand / "").is_dir():
            dirs.append(str(cand / ""))
            break
    else:
        raise RuntimeError("" "")
    ppu_sdk = os.environ.get("", "")
    if ppu_sdk:
        tgt = os.path.join(ppu_sdk, "", "", "")
        if os.path.isdir(tgt):
            dirs.insert(0, tgt)
    return dirs
]=] _expected)
  if(NOT _tokens STREQUAL _expected OR NOT _tokens_layout STREQUAL _expected_layout)
    message(FATAL_ERROR "Unsupported FFA kernel_include_dirs structure")
  endif()
  _fa_literal(_tokens_strings S0 _include)
  _fa_literal(_tokens_strings S1 _environment)
  _fa_literal(_tokens_strings S2 _default)
  _fa_literal(_tokens_strings S3 _csrc)
  _fa_literal(_tokens_strings S4 _actlize)
  _fa_literal(_tokens_strings S5 _check_include)
  _fa_literal(_tokens_strings S6 _actlize_include)
  _fa_literal(_tokens_strings S9 _sdk_environment)
  if(NOT _environment STREQUAL "ACTLIZE_DIR" OR NOT _default STREQUAL ""
      OR NOT _sdk_environment STREQUAL "PPU_SDK")
    message(FATAL_ERROR "Unsupported FFA include environment contract")
  endif()
  set(_include_path "${_module}")
  cmake_path(APPEND _include_path "${_include}")
  set(_includes "${_include_path}" "${_repo}")
  set(_candidates "")
  if(NOT "$ENV{ACTLIZE_DIR}" STREQUAL "")
    list(APPEND _candidates "$ENV{ACTLIZE_DIR}")
  endif()
  set(_local_actlize "${_repo}")
  cmake_path(APPEND _local_actlize "${_csrc}" "${_actlize}")
  list(APPEND _candidates "${_local_actlize}")
  set(_found FALSE)
  foreach(_candidate IN LISTS _candidates)
    set(_check_path "${_candidate}")
    cmake_path(APPEND _check_path "${_check_include}")
    if(IS_DIRECTORY "${_check_path}")
      set(_include_path "${_candidate}")
      cmake_path(APPEND _include_path "${_actlize_include}")
      list(APPEND _includes "${_include_path}")
      set(_found TRUE)
      break()
    endif()
  endforeach()
  if(NOT _found)
    message(FATAL_ERROR "actlize headers not found; set ACTLIZE_DIR or provide csrc/actlize")
  endif()
  # 旧 metadata 包装器移除 PPU_SDK；SDK include 由父 CMake target 注入。
  _fa_list("${_build}" NVCC_FLAGS _cuda_flags)
  _fa_list("${_build}" GENCODE _gencode)
  _fa_list("${_build}" CXX_FLAGS _cxx_flags)
  list(APPEND _cuda_flags ${_gencode})
  set(_fa_sources "${_absolute_sources}" PARENT_SCOPE)
  set(_fa_kernel_includes "${_includes}" PARENT_SCOPE)
  set(_fa_cuda_flags "${_cuda_flags}" PARENT_SCOPE)
  set(_fa_cxx_flags "${_cxx_flags}" PARENT_SCOPE)
endfunction()

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

# PPU compute_80 expands to both targets; compute_80a alone does not include sm89.
# When sm89 is included, the FFA target still emits only compute_89/sm_89.
torch_cuda_get_nvcc_gencode_flag(_fa_arch_flags)
if(NOT "${_fa_arch_flags}" MATCHES "arch=compute_(80|89),")
  message(STATUS "USE_FLEX_FLASH_ATTENTION requires an sm_89 target; skipping")
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
# Isolate global architectures and CUDA macros in the FFA subdirectory without changing torch_cuda.
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/flex_flash_attention"
                 "${CMAKE_BINARY_DIR}/flex_flash_attention")

# No target_link_libraries here on purpose: see the build-flow note at
# the top.  The call sites (attention.cpp, cuda/sdp_utils.cpp) obtain the
# entry points via flex_flash_attention_loader.h at first use.  Keep the macro
# and the build ordering (the so must exist by the time torch is complete
# and gets packaged).
add_dependencies(torch_cuda flex_flash_attention_lib)
target_compile_definitions(torch_cuda PRIVATE USE_FLEX_FLASH_ATTENTION)
target_compile_definitions(torch_cpu PRIVATE USE_FLEX_FLASH_ATTENTION)

install(TARGETS flex_flash_attention_lib
        LIBRARY DESTINATION "${TORCH_INSTALL_LIB_DIR}")

message(STATUS
        "USE_FLEX_FLASH_ATTENTION: building ${FLEX_FLASH_ATTENTION_SO} "
        "(dlopen'd at runtime, not linked into libtorch)")

