get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${extension_libs}
  MLIRFuncDialect
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRLinalgDialect
  MLIROptLib
  MLIRPass
  MLIRSparseTensorDialect
  MLIRSparseTensorTransforms
  MLIRSupport
  ElliotTransforms
)

add_llvm_executable(elliot-opt elliot-opt.cpp)
llvm_update_compile_flags(elliot-opt)
target_link_libraries(elliot-opt PRIVATE ${LIBS})
