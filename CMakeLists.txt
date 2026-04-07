cmake_minimum_required(VERSION 3.15.0)
project(elliot LANGUAGES CXX C)

set(LLVM_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/llvm" CACHE PATH "Path to LLVM cmake configs")
set(MLIR_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/mlir" CACHE PATH "Path to MLIR cmake configs")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})
