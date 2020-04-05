//===- LowerGpuOpsToNVVMOps.cpp - MLIR GPU to NVVM lowering passes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate NVVMIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToCIM/LinalgToCIMPass.h"

#include "mlir/Dialect/CIM/CIMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "mlir-linalg-to-cim"

using namespace mlir;

namespace {

struct MatmulOpLowering : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(linalg::MatmulOp op,
                                     PatternRewriter &rewriter) const final {

    auto hostA = op.getOperand(0);
    auto hostB = op.getOperand(1);
    auto hostC = op.getOperand(2);

    auto aType = hostA.getType();
    auto bType = hostB.getType();
    auto cType = hostC.getType();

    auto devA =
        rewriter.create<cim::MemcpyToDeviceOp>(op.getLoc(), aType, hostA)
            .getResult();
    auto devB =
        rewriter.create<cim::MemcpyToDeviceOp>(op.getLoc(), bType, hostB)
            .getResult();
    auto devC =
        rewriter.create<cim::MemcpyToDeviceOp>(op.getLoc(), cType, hostC)
            .getResult();

    rewriter.create<cim::MatmulOp>(op.getLoc(), devA, devB, devC);

    rewriter.create<cim::MemcpyOp>(op.getLoc(), devC, hostC, "toHost");

    rewriter.create<cim::DeallocOp>(op.getLoc(), devA);
    rewriter.create<cim::DeallocOp>(op.getLoc(), devB);
    rewriter.create<cim::DeallocOp>(op.getLoc(), devC);

    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

/// A pass that replaces Linalg operations with their corresponding CIM
/// equivalent.
class LowerLinalgOpsToCIMOpsPass
    : public OperationPass<LowerLinalgOpsToCIMOpsPass, ModuleOp> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    OwningRewritePatternList patterns;
    populateLinalgToCIMConversionPatterns(patterns, &getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<cim::CIMDialect>();
    target.addIllegalOp<linalg::MatmulOp>();

    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
  }
};
} // anonymous namespace

void mlir::populateLinalgToCIMConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<MatmulOpLowering>(ctx);
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertLinalgToCIMPass() {
  return std::make_unique<LowerLinalgOpsToCIMOpsPass>();
}

static PassRegistration<LowerLinalgOpsToCIMOpsPass>
    pass("convert-linalg-to-cim",
         "Convert the operations from the linalg dialect into the CIM dialect");
