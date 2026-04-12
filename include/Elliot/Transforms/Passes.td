#ifndef ELLIOT_PASSES
#define ELLIOT_PASSES

include "mlir/Pass/PassBase.td"

def ConvertSpMVToElliotJit : Pass<"convert-spmv-to-elliot", "::mlir::ModuleOp"> {
  let summary = "Converts CSR SpMV operations into Elliot JIT runtime calls";
  let description = [{
    This pass intercepts linalg.generic SpMV,
    extracts their underlying memref buffers, and replaces the operation with a
    function call to the Elliot NVRTC JIT runtime.
  }];
  let constructor = "elliot::createConvertSpMVToElliotJitPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::sparse_tensor::SparseTensorDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::memref::MemRefDialect"];
}

#endif // ELLIOT_PASSES
