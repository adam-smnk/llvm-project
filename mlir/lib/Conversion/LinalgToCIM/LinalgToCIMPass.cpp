//===- LinalgToCIMPass.cpp - MLIR Linalg to CIM lowering passes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Linalg operations to CIM
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToCIM/LinalgToCIMPass.h"

#include "mlir/Dialect/CIM/CIMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "mlir-linalg-to-cim"

using namespace mlir;
using namespace mlir::linalg;

namespace {

// TODO(adam-smnk) Handle case with additional sign extension operation
// that is present in case of mixed precision arguments
static bool hasMultiplyAddBody(linalg::GenericOp op) {
  auto &r = op.region();
  if (r.empty())
    return false;
  if (r.getBlocks().size() != 1)
    return false;
  auto &ops = r.front().getOperations();
  if (ops.size() != 3)
    return false;

  using mlir::matchers::m_Val;
  auto a = m_Val(r.front().getArgument(0));
  auto b = m_Val(r.front().getArgument(1));
  auto c = m_Val(r.front().getArgument(2));

  // TODO(adam-smnk) Generalize permutation matching.
  auto pattern1 = m_Op<YieldOp>(m_Op<AddIOp>(m_Op<MulIOp>(a, b), c));
  auto pattern2 = m_Op<YieldOp>(m_Op<AddIOp>(c, m_Op<MulIOp>(a, b)));
  auto pattern3 = m_Op<YieldOp>(m_Op<AddIOp>(m_Op<MulIOp>(b, a), c));
  auto pattern4 = m_Op<YieldOp>(m_Op<AddIOp>(c, m_Op<MulIOp>(b, a)));
  return pattern1.match(&ops.back()) || pattern2.match(&ops.back()) ||
         pattern3.match(&ops.back()) || pattern4.match(&ops.back());
}

static bool isGemm(linalg::GenericOp genericOp) {
  auto *ctx = genericOp.getContext();
  auto m = getAffineDimExpr(0, ctx);
  auto n = getAffineDimExpr(1, ctx);
  auto k = getAffineDimExpr(2, ctx);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}));
  auto maps = ArrayAttr::get({mapA, mapB, mapC}, ctx);
  return genericOp.getNumInputs() == 2 && genericOp.getNumOutputs() == 1 &&
         genericOp.indexing_maps() == maps && hasMultiplyAddBody(genericOp);
}

static bool isGevm(linalg::GenericOp genericOp) {
  auto *ctx = genericOp.getContext();
  auto k = getAffineDimExpr(0, ctx);
  auto n = getAffineDimExpr(1, ctx);
  auto mapA = AffineMapAttr::get(AffineMap::get(2, 0, {k}));
  auto mapB = AffineMapAttr::get(AffineMap::get(2, 0, {k, n}));
  auto mapC = AffineMapAttr::get(AffineMap::get(2, 0, {n}));
  auto maps = ArrayAttr::get({mapA, mapB, mapC}, ctx);
  return genericOp.getNumInputs() == 2 && genericOp.getNumOutputs() == 1 &&
         genericOp.indexing_maps() == maps && hasMultiplyAddBody(genericOp);
}

// TODO(adam-smnk) Extend check to handle cases with implicit transposition
// and automatically add transposition where needed
// TODO(adam-smnk) Extend check to also cover tensor contraction
static bool isMatmul(linalg::GenericOp genericOp) {
  return isGemm(genericOp) || isGevm(genericOp);
}

static void replaceOpWithCIMMatmul(Operation *op, PatternRewriter &rewriter) {
  auto matA = op->getOperand(0);
  auto matB = op->getOperand(1);
  auto matC = op->getOperand(2);

  auto cimTileID = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));

  rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), cimTileID, matB);

  // TODO(adam-smnk) Check if MatmulOp can be replaced by vector-matrix mul
  if (isa<linalg::MatmulOp>(op) || isGemm(cast<linalg::GenericOp>(op))) {
    rewriter.create<cim::GemmOp>(op->getLoc(), cimTileID, matA, matC);
  } else {
    rewriter.create<cim::GevmOp>(op->getLoc(), cimTileID, matA, matC);
  }

  rewriter.create<cim::BarrierOp>(op->getLoc(), cimTileID);

  rewriter.eraseOp(op);
}

// TODO(adam-smnk) Add support for matmul with accumulation
struct MatmulOpLowering : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(linalg::MatmulOp op,
                                     PatternRewriter &rewriter) const final {
    replaceOpWithCIMMatmul(op, rewriter);
    return matchSuccess();
  }
};

struct GenericOpLowering : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(linalg::GenericOp op,
                                     PatternRewriter &rewriter) const final {
    replaceOpWithCIMMatmul(op, rewriter);
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
    target.addLegalOp<ConstantOp>();
    target.addIllegalOp<linalg::MatmulOp>();
    target.addDynamicallyLegalOp<linalg::GenericOp>(
        [&](linalg::GenericOp op) { return !isMatmul(op); });

    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
  }
};
} // anonymous namespace

void mlir::populateLinalgToCIMConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<MatmulOpLowering, GenericOpLowering>(ctx);
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertLinalgToCIMPass() {
  return std::make_unique<LowerLinalgOpsToCIMOpsPass>();
}

static PassRegistration<LowerLinalgOpsToCIMOpsPass>
    pass("convert-linalg-to-cim",
         "Convert the operations from the linalg dialect into the CIM dialect");
