# Enable architecture pruning together with the SDK host-registration fix, not through a standalone macro.
function(torch_ppu_elementwise_launcher target)
  if(NOT USE_PPU OR NOT USE_ELEMENTWISE_OPT)
    return()
  endif()
  if(NOT Python_EXECUTABLE)
    message(FATAL_ERROR "PPU elementwise build requires Python_EXECUTABLE")
  endif()
  get_filename_component(_script
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/build/ppu_elementwise_launcher.py" ABSOLUTE)
  get_target_property(_launcher ${target} CUDA_COMPILER_LAUNCHER)
  if(NOT _launcher)
    set(_launcher "")
  endif()
  set_property(TARGET ${target} PROPERTY CUDA_COMPILER_LAUNCHER
    "${Python_EXECUTABLE};${_script};--;${_launcher}")
  # Launcher changes must also rebuild Ninja objects, not just invalidate the compiler cache.
  get_target_property(_sources ${target} SOURCES)
  foreach(_source IN LISTS _sources)
    if(_source MATCHES "\\.cu$")
      set_property(SOURCE "${_source}" APPEND PROPERTY OBJECT_DEPENDS "${_script}")
    endif()
  endforeach()
endfunction()
