#ifndef ELLIOT_PASSES
#define ELLIOT_PASSES

include "mlir/Pass/PassBase.td"

def ConvertSpMMToElliotJit : Pass<"convert-spmm-to-elliot", "ModuleOp"> {
  let summary = "Converts CSR SpMM operations into Elliot JIT runtime calls";
  let description = [{
    This pass intercepts linalg.generic sparse matrix multiplications (CSR x Dense),
    extracts their underlying memref buffers, and replaces the operation with a
    function call to the Elliot NVRTC JIT runtime.
  }];
  let constructor = "elliot::createConvertSpMMToElliotJitPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::sparse_tensor::SparseTensorDialect",
                           "mlir::memref::MemRefDialect"];
}

#endif // ELLIOT_PASSES
