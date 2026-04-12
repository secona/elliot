add_mlir_library(ElliotTransforms
  ConvertSpMVToElliotJit.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Elliot/Transforms

  DEPENDS
  ElliotPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSparseTensorDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransformUtils
)
