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

#include "mlir/Dialect/CIM/IR/CIMDialect.h"
#include "mlir/Dialect/CIM/Utils/RuntimeUtils.h"
#include "mlir/Dialect/CIM/Utils/StaticUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <set>

#define DEBUG_TYPE "mlir-linalg-to-cim"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::cim;

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

// Assumes that the genericOp is a contraction
static bool isGemm(linalg::GenericOp genericOp) {
  auto resultDims = getResultDims(genericOp);

  auto dimsA = resultDims[0];
  auto dimsB = resultDims[1];
  auto dimsC = resultDims[2];

  // C(m, n) = A(m, k) * B(k, n)
  return dimsA.size() == 2 && dimsB.size() == 2 && dimsC.size() == 2;
}

// Assumes that the genericOp is a contraction
static bool isGevm(linalg::GenericOp genericOp) {
  auto resultDims = getResultDims(genericOp);

  auto dimsA = resultDims[0];
  auto dimsB = resultDims[1];
  auto dimsC = resultDims[2];

  // C(n) = A(k) * B(k, n)
  return dimsA.size() == 1 && dimsB.size() == 2 && dimsC.size() == 1;
}

// Assumes that the genericOp is a contraction
static bool isGemv(linalg::GenericOp genericOp) {
  auto resultDims = getResultDims(genericOp);

  auto dimsA = resultDims[0];
  auto dimsB = resultDims[1];
  auto dimsC = resultDims[2];

  // C(n) = A(k) * B(k, n)
  return dimsA.size() == 2 && dimsB.size() == 1 && dimsC.size() == 1;
}

static bool isContraction(linalg::GenericOp genericOp) {
  if (!(genericOp.getNumInputs() == 2 && genericOp.getNumOutputs() == 1 &&
        hasMultiplyAddBody(genericOp))) {
    return false;
  }

  auto resultDims = getResultDims(genericOp);

  auto dimsPosA = getDimsPositions(resultDims[0]);
  auto dimsPosB = getDimsPositions(resultDims[1]);
  auto dimsPosC = getDimsPositions(resultDims[2]);

  std::set<unsigned> dimsA(dimsPosA.begin(), dimsPosA.end());
  std::set<unsigned> dimsB(dimsPosB.begin(), dimsPosB.end());
  std::set<unsigned> dimsC(dimsPosC.begin(), dimsPosC.end());

  auto uncontrDimsA = setIntersection<unsigned>(dimsA, dimsC);
  auto uncontrDimsB = setIntersection<unsigned>(dimsB, dimsC);

  // The uncontracted dimensions of either input tensor cannot
  // exceed the size of the output tensor C
  if ((uncontrDimsA.size() > dimsPosC.size()) ||
      (uncontrDimsB.size() > dimsPosC.size())) {
    return false;
  }

  auto contrDimsA = setDifference<unsigned>(dimsA, uncontrDimsA);
  auto contrDimsB = setDifference<unsigned>(dimsB, uncontrDimsB);
  auto contrDims = setUnion<unsigned>(contrDimsA, contrDimsB);

  // Check if the remaining uncontracted dimensions match those of C
  auto outputDims = setUnion<unsigned>(uncontrDimsA, uncontrDimsB);

  // Check if the uncontracted dimensions from A match the left
  // part of C and if the uncontracted dimensions from B match
  // the remaining right part of C
  std::set<unsigned> cDimsFromA(dimsPosC.begin(),
                                dimsPosC.begin() + uncontrDimsA.size());
  std::set<unsigned> cDimsFromB(dimsPosC.begin() + uncontrDimsA.size(),
                                dimsPosC.end());

  return contrDims.size() > 0 && contrDimsA == contrDimsB &&
         dimsC.size() == (uncontrDimsA.size() + uncontrDimsB.size()) &&
         outputDims == dimsC && uncontrDimsA == cDimsFromA &&
         uncontrDimsB == cDimsFromB;
}

static void createCIMContractionOp(Operation *op, PatternRewriter &rewriter,
                                   ConstantOp &tileId) {
  auto genOp = cast<GenericOp>(op);
  auto *ctx = genOp.getContext();

  auto matA = op->getOperand(0);
  auto matB = op->getOperand(1);
  auto matC = op->getOperand(2);

  auto resultMaps = getResultMaps(genOp);

  AffineMap mapA = resultMaps[0];
  AffineMap mapB = resultMaps[1];
  AffineMap mapC = resultMaps[2];

  ArrayRef<AffineExpr> dimsA = mapA.getResults();
  ArrayRef<AffineExpr> dimsB = mapB.getResults();
  ArrayRef<AffineExpr> dimsC = mapC.getResults();

  std::vector<unsigned> dimsPosA = getDimsPositions(dimsA);
  std::vector<unsigned> dimsPosB = getDimsPositions(dimsB);
  std::vector<unsigned> dimsPosC = getDimsPositions(dimsC);

  std::set<unsigned> dimsSetA(dimsPosA.begin(), dimsPosA.end());
  std::set<unsigned> dimsSetB(dimsPosB.begin(), dimsPosB.end());
  std::set<unsigned> dimsSetC(dimsPosC.begin(), dimsPosC.end());

  auto uncontrDimsA = setIntersection<unsigned>(dimsSetA, dimsSetC);
  auto uncontrDimsB = setIntersection<unsigned>(dimsSetB, dimsSetC);
  auto contractionDims = setIntersection<unsigned>(dimsSetA, dimsSetB);

  SmallVector<AffineExpr, 8U> reqLeftDimsA;
  for (unsigned i = 0; i < uncontrDimsA.size(); ++i) {
    reqLeftDimsA.push_back(getAffineDimExpr(dimsPosC[i], ctx));
  }
  SmallVector<AffineExpr, 8U> reqRightDimsA;
  for (const auto &pos : contractionDims) {
    reqRightDimsA.push_back(getAffineDimExpr(pos, ctx));
  }
  auto leftDimsA =
      AffineMapAttr::get(AffineMap::get(mapA.getNumInputs(), 0, reqLeftDimsA));
  auto rightDimsA =
      AffineMapAttr::get(AffineMap::get(mapA.getNumInputs(), 0, reqRightDimsA));

  // Flatten tensor to matrix
  Value flatA = groupDimensions(op, rewriter, matA, mapA,
                                ArrayAttr::get({leftDimsA, rightDimsA}, ctx));

  SmallVector<AffineExpr, 8U> reqLeftDimsB;
  for (const auto &pos : contractionDims) {
    reqLeftDimsB.push_back(getAffineDimExpr(pos, ctx));
  }
  SmallVector<AffineExpr, 8U> reqRightDimsB;
  for (unsigned i = 0; i < uncontrDimsB.size(); ++i) {
    unsigned cPos = uncontrDimsA.size() + i;
    reqRightDimsB.push_back(getAffineDimExpr(dimsPosC[cPos], ctx));
  }
  auto leftDimsB =
      AffineMapAttr::get(AffineMap::get(mapB.getNumInputs(), 0, reqLeftDimsB));
  auto rightDimsB =
      AffineMapAttr::get(AffineMap::get(mapB.getNumInputs(), 0, reqRightDimsB));

  // Flatten tensor to matrix
  Value flatB = groupDimensions(op, rewriter, matB, mapB,
                                ArrayAttr::get({leftDimsB, rightDimsB}, ctx));

  auto leftDimsC = AffineMapAttr::get(
      AffineMap::get(mapC.getNumInputs(), 0,
                     ArrayRef<AffineExpr>(
                         dimsC.begin(), dimsC.begin() + uncontrDimsA.size())));
  auto rightDimsC = AffineMapAttr::get(AffineMap::get(
      mapC.getNumInputs(), 0,
      ArrayRef<AffineExpr>(dimsC.begin() + uncontrDimsA.size(), dimsC.end())));

  // Flatten tensor to matrix
  Value flatC = groupDimensions(op, rewriter, matC, mapC,
                                ArrayAttr::get({leftDimsC, rightDimsC}, ctx));

  rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), tileId, flatB);
  rewriter.create<cim::GemmOp>(op->getLoc(), tileId, flatA, flatC);
  rewriter.create<cim::BarrierOp>(op->getLoc(), tileId);

  // Unflatten the contraction result and copy to the output tensor
  reshapeCopy(op, rewriter, flatC, matC,
              ArrayAttr::get({leftDimsC, rightDimsC}, ctx),
              ArrayRef<unsigned>(), true);
}

static void createCIMGemmOp(Operation *op, PatternRewriter &rewriter,
                            ConstantOp &tileId) {
  auto genOp = cast<GenericOp>(op);
  auto *ctx = genOp.getContext();

  auto matA = op->getOperand(0);
  auto matB = op->getOperand(1);
  auto matC = op->getOperand(2);

  auto resultMaps = getResultMaps(genOp);

  AffineMap mapA = resultMaps[0];
  AffineMap mapB = resultMaps[1];
  AffineMap mapC = resultMaps[2];

  ArrayRef<AffineExpr> dimsA = mapA.getResults();
  ArrayRef<AffineExpr> dimsB = mapB.getResults();
  ArrayRef<AffineExpr> dimsC = mapC.getResults();

  std::vector<unsigned> dimsPosA = getDimsPositions(dimsA);
  std::vector<unsigned> dimsPosB = getDimsPositions(dimsB);
  std::vector<unsigned> dimsPosC = getDimsPositions(dimsC);

  std::set<unsigned> dimsSetA(dimsPosA.begin(), dimsPosA.end());
  std::set<unsigned> dimsSetB(dimsPosB.begin(), dimsPosB.end());
  std::set<unsigned> dimsSetC(dimsPosC.begin(), dimsPosC.end());

  auto uncontrDimsA = setIntersection<unsigned>(dimsSetA, dimsSetC);
  auto uncontrDimsB = setIntersection<unsigned>(dimsSetB, dimsSetC);
  auto contractionDims = setIntersection<unsigned>(dimsSetA, dimsSetB);

  auto transposeReqs = checkContractionTransposes(dimsPosA, dimsPosB, dimsPosC);

  Value inputB = matB;

  if (transposeReqs.transposeB) {
    SmallVector<AffineExpr, 8U> reqDimsB;
    for (const auto &pos : contractionDims) {
      reqDimsB.push_back(getAffineDimExpr(pos, ctx));
    }
    for (unsigned i = 0; i < uncontrDimsB.size(); ++i) {
      unsigned cPos = uncontrDimsA.size() + i;
      reqDimsB.push_back(getAffineDimExpr(dimsPosC[cPos], ctx));
    }
    auto transposeMap =
        AffineMap::get(mapB.getNumInputs(), 0, ArrayRef<AffineExpr>(reqDimsB));

    inputB = transposeMemRef(op, rewriter, matB, mapB, transposeMap);
  }

  rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), tileId, inputB);

  Value inputA = matA;
  Value outputC = allocateDuplicate(op, rewriter, matC);

  if (transposeReqs.transposeA) {
    SmallVector<AffineExpr, 8U> reqDimsA;
    for (unsigned i = 0; i < uncontrDimsA.size(); ++i) {
      reqDimsA.push_back(getAffineDimExpr(dimsPosC[i], ctx));
    }
    for (const auto &pos : contractionDims) {
      reqDimsA.push_back(getAffineDimExpr(pos, ctx));
    }
    auto transposeMap =
        AffineMap::get(mapA.getNumInputs(), 0, ArrayRef<AffineExpr>(reqDimsA));

    inputA = transposeMemRef(op, rewriter, matA, mapA, transposeMap);
  }

  rewriter.create<cim::GemmOp>(op->getLoc(), tileId, inputA, outputC);
  rewriter.create<cim::BarrierOp>(op->getLoc(), tileId);

  elementwiseAddition(op, rewriter, outputC, matC);
}

static void createCIMGevmOp(Operation *op, PatternRewriter &rewriter,
                            ConstantOp &tileId) {
  auto genOp = cast<GenericOp>(op);

  auto matA = op->getOperand(0);
  auto matB = op->getOperand(1);
  auto matC = op->getOperand(2);

  auto resultMaps = getResultMaps(genOp);

  auto mapB = resultMaps[1];

  auto dimsA = resultMaps[0].getResults();
  auto dimsC = resultMaps[2].getResults();

  SmallVector<AffineExpr, 8U> reqDimsB;
  for (const auto &dim : dimsA) {
    reqDimsB.push_back(dim);
  }
  for (const auto &dim : dimsC) {
    reqDimsB.push_back(dim);
  }
  auto transposeMap = AffineMap::get(2, 0, ArrayRef<AffineExpr>(reqDimsB));

  Value inputB = matB;
  Value outputC = allocateDuplicate(op, rewriter, matC);

  // Check if B needs to be transposed
  if (mapB != transposeMap) {
    inputB = transposeMemRef(op, rewriter, matB, mapB, transposeMap);
  }

  rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), tileId, inputB);
  rewriter.create<cim::GevmOp>(op->getLoc(), tileId, matA, outputC);
  rewriter.create<cim::BarrierOp>(op->getLoc(), tileId);

  elementwiseAddition(op, rewriter, outputC, matC);
}

static void createCIMGemvOp(Operation *op, PatternRewriter &rewriter,
                            ConstantOp &tileId) {
  auto genOp = cast<GenericOp>(op);
  auto ctx = op->getContext();

  auto matA = op->getOperand(0);
  auto matB = op->getOperand(1);

  auto resultMaps = getResultMaps(genOp);

  auto mapA = resultMaps[0];
  auto mapB = resultMaps[1];
  auto mapC = resultMaps[2];

  auto dimsA = mapA.getResults();

  // Convert GEMV to GEVM through tranposition:
  // (A*B)^T = B^T * A^T
  // B and C don't need to be tranposed as they are represented by
  // one dimensional vectors
  auto aTranposeMap = AffineMap::get(2, 0, {dimsA[1], dimsA[0]});
  Value matAt = transposeMemRef(op, rewriter, matA, mapA, aTranposeMap);

  // Remap operator indexing
  auto indexingMapsAttr = ArrayAttr::get({AffineMapAttr::get(mapB),
                                          AffineMapAttr::get(aTranposeMap),
                                          AffineMapAttr::get(mapC)},
                                         ctx);
  op->setAttr(getIndexingMapsAttrName(), indexingMapsAttr);

  // Rearrange operands to fit GEVM op
  op->setOperand(0, matB);
  op->setOperand(1, matAt);

  createCIMGevmOp(op, rewriter, tileId);
}

// TODO(adam-smnk) Make lowering more progressive, move some of the conversions
// into CIMtoSTD pass
static void replaceOpWithCIMMatmul(Operation *op, PatternRewriter &rewriter) {
  auto matA = op->getOperand(0);
  auto matB = op->getOperand(1);
  auto matC = op->getOperand(2);

  auto cimTileID = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));

  // TODO(adam-smnk) Check if MatmulOp/GEMM can be replaced by vector-matrix mul
  if (isa<linalg::MatmulOp>(op)) {
    // Assumes that linalg.matmul always has correct values and memrefs
    rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), cimTileID, matB);
    rewriter.create<cim::GemmOp>(op->getLoc(), cimTileID, matA, matC);
    rewriter.create<cim::BarrierOp>(op->getLoc(), cimTileID);
  } else if (isGevm(cast<linalg::GenericOp>(op))) {
    createCIMGevmOp(op, rewriter, cimTileID);
  } else if (isGemm(cast<linalg::GenericOp>(op))) {
    createCIMGemmOp(op, rewriter, cimTileID);
  } else if (isGemv(cast<linalg::GenericOp>(op))) {
    createCIMGemvOp(op, rewriter, cimTileID);
  } else {
    createCIMContractionOp(op, rewriter, cimTileID);
  }

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

// TODO(adam-smnk) Check for contiguous memory
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
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<loop::LoopOpsDialect>();
    target.addIllegalOp<linalg::MatmulOp>();
    target.addDynamicallyLegalOp<linalg::GenericOp>(
        [&](linalg::GenericOp op) { return !isContraction(op); });

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
