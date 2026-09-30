# Pre-build step previously handled by setup.py:
#   Git submodule URL sync, initialization and verification.
#
# This file is included early in CMakeLists.txt, right after
# EnvVarForwarding.cmake and before the option() declarations.  It relies on
# env vars (forwarded by EnvVarForwarding.cmake) and on CMake -D cache
# variables, both of which are available at this point.
#
# Backported from the module of the same name on the v2.13 branch, with three
# deliberate differences:
#
#  1. `git submodule sync --recursive` runs before the update.  A persisted CI
#     workspace can still carry an obsolete submodule URL in its .git/config,
#     in which case a plain update keeps fetching from the wrong remote and the
#     stale checkout is never repaired.
#  2. The update is run with --force, so a stale but non-empty submodule
#     checkout is realigned to the commit recorded in the superproject index.
#  3. The trigger is "some submodule is not at its recorded commit" instead of
#     v2.13's "every submodule directory is empty".  setup.py::
#     check_submodules() only auto-initializes when *all* submodule directories
#     are empty, so a workspace holding an older checkout of third_party/cutlass
#     is never repaired by it - which is how
#     cutlass/gemm/warp/mma_tensor_op_tile_iterator_ppu0010.h went missing at
#     compile time.  The mismatch check below is what makes this module
#     effective, and it is a no-op on a tree that is already at the recorded
#     commits.
#
# The NCCL checkout section of the v2.13 module is intentionally not
# backported: this branch builds against the NCCL shipped in the PPU SDK
# (setup.py exports USE_SYSTEM_NCCL=1) and third_party/nccl is not part of its
# build flow, so cloning NCCL from github.com at configure time would only add
# an unwanted network dependency.

find_package(Git QUIET)

# --- Submodule sync, initialization and verification ---
# Matches (and repairs the short-circuit of) setup.py::check_submodules().
if(NOT DEFINED USE_SYSTEM_LIBS OR NOT USE_SYSTEM_LIBS)
  # Read submodule paths from .gitmodules if available, otherwise use defaults.
  set(_gitmodules_file "${PROJECT_SOURCE_DIR}/.gitmodules")
  if(EXISTS "${_gitmodules_file}")
    file(STRINGS "${_gitmodules_file}" _gitmodule_lines REGEX "^[ \t]*path")
    set(_submodule_folders)
    foreach(_line IN LISTS _gitmodule_lines)
      string(REGEX REPLACE ".*=[ \t]*" "" _path "${_line}")
      list(APPEND _submodule_folders "${PROJECT_SOURCE_DIR}/${_path}")
    endforeach()
  else()
    set(_submodule_folders
      "${PROJECT_SOURCE_DIR}/third_party/gloo"
      "${PROJECT_SOURCE_DIR}/third_party/cpuinfo"
      "${PROJECT_SOURCE_DIR}/third_party/onnx"
      "${PROJECT_SOURCE_DIR}/third_party/fbgemm"
      "${PROJECT_SOURCE_DIR}/third_party/cutlass"
    )
  endif()

  set(_all_missing TRUE)
  foreach(_dir IN LISTS _submodule_folders)
    if(EXISTS "${_dir}" AND IS_DIRECTORY "${_dir}")
      file(GLOB _contents "${_dir}/*")
      if(_contents)
        set(_all_missing FALSE)
        break()
      endif()
    endif()
  endforeach()

  # Only run `git submodule` when building from a git checkout.  Source
  # tarballs / nightly build trees have no .git directory; running
  # `git submodule` there would fail and abort the build even when the
  # submodule trees are already populated from the tarball.
  set(_is_git_checkout FALSE)
  if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    set(_is_git_checkout TRUE)
  endif()

  # Detect submodules that are not initialized ('-'), that are checked out at
  # a commit different from the one recorded in the superproject index ('+'),
  # or that have merge conflicts ('U').  `git submodule status` does not flag a
  # dirty work tree, so the in-place third-party conversions applied by
  # setup.py are not mistaken for a mismatch.  Nested submodules are excluded
  # from the detection on purpose: the --recursive update below still repairs
  # them, but a nested entry must never be able to trigger a forced checkout of
  # the top-level trees.
  set(_out_of_sync)
  set(_out_of_sync_count 0)
  if(_is_git_checkout)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule status
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE _submodule_status
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _status_result
    )
    if(_status_result EQUAL 0)
      string(REPLACE "\n" ";" _status_lines "${_submodule_status}")
      foreach(_line IN LISTS _status_lines)
        if(_line MATCHES "^[+-U]")
          string(STRIP "${_line}" _line)
          list(APPEND _out_of_sync "${_line}")
        endif()
      endforeach()
      list(LENGTH _out_of_sync _out_of_sync_count)
    else()
      message(WARNING
        "git submodule status failed (exit code ${_status_result}); "
        "skipping the submodule consistency check."
      )
    endif()
  endif()

  # Remember whether the in-place third-party conversions (cudafy) have already
  # been applied.  Their stamps live inside the submodule work trees, so a
  # forced checkout would reset the converted sources while leaving the stamps
  # behind, and the next build would then skip converting an unconverted tree.
  set(_cudafy_stamp_dirs
    "${PROJECT_SOURCE_DIR}/third_party/cutlass/include/.cudafy-for-sail"
    "${PROJECT_SOURCE_DIR}/third_party/flash-attention/.cudafy-for-sail"
  )
  set(_cudafy_stamped FALSE)
  foreach(_stamp_dir IN LISTS _cudafy_stamp_dirs)
    if(EXISTS "${_stamp_dir}")
      set(_cudafy_stamped TRUE)
    endif()
  endforeach()

  if((_all_missing OR _out_of_sync_count GREATER 0) AND _is_git_checkout)
    if(_out_of_sync_count GREATER 0)
      message(STATUS "Submodules are not at the commits recorded by the superproject:")
      foreach(_entry IN LISTS _out_of_sync)
        message(STATUS "  ${_entry}")
      endforeach()
    else()
      message(STATUS "Initializing git submodules...")
    endif()

    # Refresh the submodule URLs in .git/config from .gitmodules first, so that
    # the update below cannot fetch from an obsolete remote left behind by an
    # earlier revision of .gitmodules.
    message(STATUS "Syncing submodule URLs (git submodule sync --recursive)...")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule sync --recursive
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      RESULT_VARIABLE _sync_result
    )
    if(NOT _sync_result EQUAL 0)
      message(FATAL_ERROR
        "Git submodule sync failed (exit code ${_sync_result}). Please run:\n"
        "  git submodule sync --recursive"
      )
    endif()

    message(STATUS "Updating submodules (git submodule update --init --recursive --force)...")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive --force
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      RESULT_VARIABLE _submodule_result
    )
    if(NOT _submodule_result EQUAL 0)
      message(FATAL_ERROR
        "Git submodule initialization failed (exit code ${_submodule_result}). "
        "Please run:\n"
        "  git submodule sync --recursive\n"
        "  git submodule update --init --recursive --force"
      )
    endif()

    if(_cudafy_stamped)
      # setup.py converts third_party/flash-attention and third_party/cutlass
      # in place *before* cmake is invoked, so the forced checkout above has
      # just discarded those conversions.  Drop the now stale stamps and stop
      # here: the next build re-applies the conversions on top of the correct
      # checkouts, exactly as it would on a fresh clone.
      foreach(_stamp_dir IN LISTS _cudafy_stamp_dirs)
        if(EXISTS "${_stamp_dir}")
          message(STATUS "Removing stale cudafy stamp directory: ${_stamp_dir}")
          file(REMOVE_RECURSE "${_stamp_dir}")
        endif()
      endforeach()
      message(FATAL_ERROR
        "Submodules were realigned to the commits recorded by the superproject.\n"
        "The in-place third-party conversions applied earlier in this run by\n"
        "setup.py (cudafy for third_party/flash-attention and\n"
        "third_party/cutlass) live inside those submodule work trees and were\n"
        "discarded by the forced checkout; their stamps have been removed too.\n"
        "Re-run the build so that the conversions are applied on top of the\n"
        "correct checkouts."
      )
    endif()
  endif()

  # Verify submodules contain expected files (catches corrupt/partial checkouts).
  set(_expected_files CMakeLists.txt Makefile setup.py LICENSE LICENSE.md LICENSE.txt)
  foreach(_dir IN LISTS _submodule_folders)
    set(_found FALSE)
    foreach(_file IN LISTS _expected_files)
      if(EXISTS "${_dir}/${_file}")
        set(_found TRUE)
        break()
      endif()
    endforeach()
    if(NOT _found)
      message(FATAL_ERROR
        "Submodule ${_dir} appears incomplete (none of "
        "${_expected_files} found).\n"
        "Please run: git submodule sync --recursive && "
        "git submodule update --init --recursive --force"
      )
    endif()
  endforeach()
  # Extra check for fbgemm's nested dependency
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/third_party/fbgemm/external/asmjit/CMakeLists.txt")
    message(FATAL_ERROR
      "third_party/fbgemm/external/asmjit appears incomplete.\n"
      "Please run: git submodule sync --recursive && "
      "git submodule update --init --recursive --force"
    )
  endif()
endif()
