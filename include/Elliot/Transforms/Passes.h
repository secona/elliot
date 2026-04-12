#ifndef ELLIOT_TRANSFORMS_PASSES_H
#define ELLIOT_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DECL
#include "Elliot/Transforms/Passes.h.inc"
} // namespace mlir

namespace elliot {

std::unique_ptr<mlir::Pass> createConvertSpMVToElliotJitPass();

#define GEN_PASS_REGISTRATION
#include "Elliot/Transforms/Passes.h.inc"
#undef GEN_PASS_REGISTRATION

} // namespace elliot

#endif // ELLIOT_TRANSFORMS_PASSES_H
