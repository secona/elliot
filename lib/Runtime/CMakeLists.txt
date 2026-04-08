add_library(ElliotRuntime SHARED
  ElliotRuntime.cpp
)

target_link_libraries(ElliotRuntime
  PRIVATE
  CUDA::cuda_driver
  CUDA::nvrtc
)

set_target_properties(ElliotRuntime PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
