#include "Elliot/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DEF_CONVERTSPMMTOELLIOTJIT
#include "Elliot/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace elliot {
namespace {

struct SpMMToElliotRewrite : public OpRewritePattern<linalg::GenericOp> {
  SpMMToElliotRewrite(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
      return failure();

    Value lhs = op.getDpsInputs()[0];
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());

    if (!lhsType || !getSparseTensorEncoding(lhsType))
      return failure();

    // TODO: Add stricter check to ensure encoding is specifically CSR

    Location loc = op.getLoc();

    Type indexType = rewriter.getIndexType();
    Type ptrIdxMemRefType = MemRefType::get({ShapedType::kDynamic}, indexType);
    Type elementType = lhsType.getElementType();
    Type valMemRefType = MemRefType::get({ShapedType::kDynamic}, elementType);

    Value pointers = ToPositionsOp::create(rewriter, loc, ptrIdxMemRefType,
                                             lhs, rewriter.getIndexAttr(1));
    Value indices = ToCoordinatesOp::create(rewriter, loc, ptrIdxMemRefType,
                                            lhs, rewriter.getIndexAttr(1));
    Value values = ToValuesOp::create(rewriter, loc, valMemRefType, lhs);

    Value rhs = op.getDpsInputs()[1];
    Value out = op.getDpsInits()[0];

    ModuleOp module = op->getParentOfType<ModuleOp>();
    StringRef jitFuncName = "elliot_jit_spmv_csr_f32";

    if (!module.lookupSymbol<func::FuncOp>(jitFuncName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto funcType = rewriter.getFunctionType(
          {pointers.getType(), indices.getType(), values.getType(),
           rhs.getType(), out.getType()},
          {});
      func::FuncOp jitFunc =
          func::FuncOp::create(rewriter, loc, jitFuncName, funcType);
      jitFunc.setPrivate();
    }

    func::CallOp::create(rewriter, loc, jitFuncName, TypeRange(),
                         ValueRange{pointers, indices, values, rhs, out});

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ConvertSpMMToElliotJitPass
    : public mlir::impl::ConvertSpMMToElliotJitBase<
          ConvertSpMMToElliotJitPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<SpMMToElliotRewrite>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createConvertSpMMToElliotJitPass() {
  return std::make_unique<ConvertSpMMToElliotJitPass>();
}

} // namespace elliot
