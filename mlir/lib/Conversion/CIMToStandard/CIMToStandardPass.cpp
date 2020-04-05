//===- CIMToStandardPass.cpp - MLIR CIM to Standard lowering passes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate Standard operations for CIM dialect
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/CIMToStandard/CIMToStandardPass.h"

#include "mlir/Dialect/CIM/CIMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "mlir-cim-to-standard"

using namespace mlir;
using namespace mlir::cim;

namespace {

// Get a SymbolRefAttr containing the library function name for the CimOp.
// If the library function does not exist, insert a declaration.
template <typename CimOp>
static FlatSymbolRefAttr getLibraryCallSymbolRef(Operation *op,
                                                 PatternRewriter &rewriter) {
  auto cimOp = cast<CimOp>(op);
  auto fnName = cimOp.getLibraryCallName();
  if (fnName.empty()) {
    op->emitWarning("No library call defined for: ") << *op;
    return {};
  }

  // fnName is a dynamic std::String, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr = rewriter.getSymbolRefAttr(fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnName)) {
    return fnNameAttr;
  }

  SmallVector<Type, 4> inputTypes(op->getOperandTypes());
  auto libFnType = FunctionType::get(inputTypes, op->getResultTypes(),
                                     rewriter.getContext());

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  rewriter.create<FuncOp>(op->getLoc(), fnNameAttr.getValue(), libFnType,
                          ArrayRef<NamedAttribute>{});
  return fnNameAttr;
}

// CimOpConversion<CimOp> creates a new call to the
// `CimOp::getLibraryCallName()` function.
// The implementation of the function can be either in the same module or in an
// externally linked library.
template <typename CimOp>
class CimOpConversion : public OpRewritePattern<CimOp> {
public:
  using OpRewritePattern<CimOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(CimOp op,
                                     PatternRewriter &rewriter) const override {
    auto libraryCallName = getLibraryCallSymbolRef<CimOp>(op, rewriter);
    if (!libraryCallName)
      return this->matchFailure();

    Operation *cimOp = op;
    rewriter.replaceOpWithNewOp<mlir::CallOp>(cimOp, libraryCallName.getValue(),
                                              cimOp->getResultTypes(),
                                              cimOp->getOperands());

    return this->matchSuccess();
  }
};

/// A pass that replaces Linalg operations with their corresponding CIM
/// equivalent.
class LowerCIMOpsToStandardOpsPass
    : public OperationPass<LowerCIMOpsToStandardOpsPass, ModuleOp> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    OwningRewritePatternList patterns;
    populateCIMToStandardConversionPatterns(patterns, &getContext());

    ConversionTarget target(getContext());
    target.addIllegalDialect<cim::CIMDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<FuncOp>();

    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
  }
};
} // anonymous namespace

void mlir::populateCIMToStandardConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns
      .insert<CimOpConversion<cim::AllocOp>, CimOpConversion<cim::DeallocOp>,
              CimOpConversion<cim::MatmulOp>, CimOpConversion<cim::MemcpyOp>,
              CimOpConversion<cim::MemcpyToDeviceOp>,
              CimOpConversion<cim::MemcpyToHostOp>>(ctx);
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertCIMToStandardPass() {
  return std::make_unique<LowerCIMOpsToStandardOpsPass>();
}

static PassRegistration<LowerCIMOpsToStandardOpsPass> pass(
    "convert-cim-to-std",
    "Convert the operations from the CIM dialect into the Standard dialect");
