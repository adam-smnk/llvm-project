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
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <set>

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

static auto getResultMaps(linalg::GenericOp genericOp) {
  auto mapsRange = genericOp.indexing_maps().getAsRange<AffineMapAttr>();
  return functional::map([](AffineMapAttr a) { return a.getValue(); },
                         mapsRange);
}

static auto getResultDims(linalg::GenericOp genericOp) {
  auto mapsRange = genericOp.indexing_maps().getAsRange<AffineMapAttr>();
  return functional::map(
      [](AffineMapAttr a) { return a.getValue().getResults(); }, mapsRange);
}

static bool isGemm(linalg::GenericOp genericOp) {
  auto *ctx = genericOp.getContext();
  // TODO(adam-smnk) Generalize dimension order (fails when the order is
  // swapped).
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
  auto resultDims = getResultDims(genericOp);

  auto dimsA = resultDims[0];
  auto dimsB = resultDims[1];
  auto dimsC = resultDims[2];

  // C(n) = A(k) * B(k, n)
  return dimsA.size() == 1 && dimsB.size() == 2 && dimsC.size() == 1;
}

static std::vector<unsigned>
getDimsPositions(const ArrayRef<AffineExpr> &affineDims) {

  std::vector<unsigned> dims;

  for (auto dim : affineDims) {
    if (dim.getKind() == AffineExprKind::DimId) {
      dims.push_back(dim.cast<AffineDimExpr>().getPosition());
    }
  }

  return dims;
}

template <typename T>
std::set<T> setIntersection(const std::set<T> &setA, const std::set<T> &setB) {
  std::set<unsigned int> intersectionSet;
  std::set_intersection(
      setA.begin(), setA.end(), setB.begin(), setB.end(),
      std::inserter(intersectionSet, intersectionSet.begin()));

  return intersectionSet;
}

template <typename T>
std::set<T> setDifference(const std::set<T> &setA, const std::set<T> &setB) {
  std::set<unsigned int> diffSet;
  std::set_difference(setA.begin(), setA.end(), setB.begin(), setB.end(),
                      std::inserter(diffSet, diffSet.begin()));

  return diffSet;
}

template <typename T>
std::set<T> setUnion(const std::set<T> &setA, const std::set<T> &setB) {
  std::set<unsigned int> unionSet;
  std::set_union(setA.begin(), setA.end(), setB.begin(), setB.end(),
                 std::inserter(unionSet, unionSet.begin()));

  return unionSet;
}

static bool isContraction(linalg::GenericOp genericOp) {
  if (!(genericOp.getNumInputs() == 2 && genericOp.getNumOutputs() == 1 &&
        hasMultiplyAddBody(genericOp))) {
    llvm::errs() << "Invalid Op body"
                 << "\n";
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

  auto outputDims = setUnion<unsigned>(uncontrDimsA, uncontrDimsB);

  return outputDims == dimsC;
}

struct TransposeAnalysisResults {
  bool transposeA;
  bool transposeB;

  TransposeAnalysisResults() : TransposeAnalysisResults(false, false){};
  TransposeAnalysisResults(bool transposeA_, bool transposeB_)
      : transposeA(transposeA_), transposeB(transposeB_){};
};

// TODO(adam-smnk) Consider splitting need to transpose contracted and
// uncontracted dimensions.
static TransposeAnalysisResults
checkContractionTransposes(const std::vector<unsigned> &dimsPosA,
                           const std::vector<unsigned> &dimsPosB,
                           const std::vector<unsigned> &dimsPosC) {
  std::set<unsigned> dimsSetA(dimsPosA.begin(), dimsPosA.end());
  std::set<unsigned> dimsSetB(dimsPosB.begin(), dimsPosB.end());
  std::set<unsigned> dimsSetC(dimsPosC.begin(), dimsPosC.end());

  auto uncontrDimsA = setIntersection<unsigned>(dimsSetA, dimsSetC);
  auto uncontrDimsB = setIntersection<unsigned>(dimsSetB, dimsSetC);
  auto contractionDims = setIntersection<unsigned>(dimsSetA, dimsSetB);

  // check if A and B dimensions match
  for (unsigned i = 0; i < contractionDims.size(); ++i) {
    unsigned aPos = uncontrDimsA.size() + i;

    if (dimsPosA[aPos] != dimsPosB[i]) {
      return TransposeAnalysisResults(true, true);
    }
  }

  TransposeAnalysisResults reqs(false, false);

  // check if A and C dimensions match
  for (unsigned i = 0; i < uncontrDimsA.size(); ++i) {
    if (dimsPosA[i] != dimsPosC[i]) {
      reqs.transposeA = true;
      break;
    }
  }

  // check if B and C dimensions match
  for (unsigned i = 0; i < uncontrDimsB.size(); ++i) {
    unsigned bPos = contractionDims.size() + i;
    unsigned cPos = uncontrDimsA.size() + i;

    if (dimsPosB[bPos] != dimsPosC[cPos]) {
      reqs.transposeB = true;
      break;
    }
  }

  return reqs;
}

static Value transposeMemRef(Operation *op, PatternRewriter &rewriter,
                             const Value &memRef, const AffineMap &memRefMap,
                             const AffineMap &targetMap) {
  auto *ctx = cast<GenericOp>(op).getContext();
  Value transposedMemory = memRef;

  auto memRefDimsPos = getDimsPositions(memRefMap.getResults());
  auto reqDimsPos = getDimsPositions(targetMap.getResults());
  auto memRefType = memRef.getType().cast<MemRefType>();

  // Map the original memref to its own order and make output permutation
  // match post-transposition order
  SmallVector<AffineExpr, 8U> inputPermutation;
  SmallVector<AffineExpr, 8U> outputPermutation;
  for (unsigned i = 0; i < reqDimsPos.size(); ++i) {
    inputPermutation.push_back(getAffineDimExpr(i, ctx));

    auto it =
        std::find(memRefDimsPos.begin(), memRefDimsPos.end(), reqDimsPos[i]);
    int pos = std::distance(memRefDimsPos.begin(), it);
    outputPermutation.push_back(getAffineDimExpr(pos, ctx));
  }

  auto inputPermutationMap = AffineMap::get(
      memRefMap.getNumResults(), 0, ArrayRef<AffineExpr>(inputPermutation));
  auto outputPermutationMap = AffineMap::get(
      memRefMap.getNumResults(), 0, ArrayRef<AffineExpr>(outputPermutation));

  // Check if there are any memory layout changes required
  if (inputPermutation != outputPermutation) {
    ArrayRef<int64_t> memShape = memRefType.getShape();
    SmallVector<int64_t, 8U> transposedShape;
    for (auto dimPos : reqDimsPos) {
      auto it = std::find(memRefDimsPos.begin(), memRefDimsPos.end(), dimPos);
      int i = std::distance(memRefDimsPos.begin(), it);
      transposedShape.push_back(memShape[i]);
    }
    MemRefType transposedMemRef =
        MemRefType::Builder(memRefType).setShape(transposedShape);

    // Allocate new memref for tranposed data.
    // Get dynamic memory sizes of the original memref
    // and put them in post-tranposition order.
    auto outputPermutationPos =
        getDimsPositions(ArrayRef<AffineExpr>(outputPermutation));

    SmallVector<Value, 8U> allocOperands;
    auto aMemRefShape = memRefType.getShape();
    for (unsigned i = 0; i < aMemRefShape.size(); ++i) {
      if (aMemRefShape[i] == -1) {
        allocOperands.push_back(rewriter.create<DimOp>(
            op->getLoc(), memRef, outputPermutationPos[i]));
      }
    }
    transposedMemory =
        rewriter.create<AllocOp>(op->getLoc(), transposedMemRef, allocOperands)
            .getResult();

    rewriter.create<linalg::CopyOp>(op->getLoc(), memRef, transposedMemory,
                                    AffineMapAttr::get(inputPermutationMap),
                                    AffineMapAttr::get(outputPermutationMap));

    // Deallocate transposed data at the end of block after CIM computations are
    // finished
    auto deallocOp = rewriter.create<DeallocOp>(op->getLoc(), transposedMemory)
                         .getOperation();
    deallocOp->moveBefore(&(deallocOp->getBlock()->back()));
  }

  return transposedMemory;
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

  if (transposeReqs.transposeB) {
    SmallVector<AffineExpr, 8U> reqDimsB;
    for (auto pos : contractionDims) {
      reqDimsB.push_back(getAffineDimExpr(pos, ctx));
    }
    for (unsigned i = 0; i < uncontrDimsB.size(); ++i) {
      unsigned cPos = uncontrDimsA.size() + i;
      reqDimsB.push_back(getAffineDimExpr(dimsPosC[cPos], ctx));
    }
    auto transposeMap =
        AffineMap::get(mapB.getNumInputs(), 0, ArrayRef<AffineExpr>(reqDimsB));

    Value transposedB = transposeMemRef(op, rewriter, matB, mapB, transposeMap);

    rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), tileId, transposedB);
  } else {
    rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), tileId, matB);
  }

  if (transposeReqs.transposeA) {
    SmallVector<AffineExpr, 8U> reqDimsA;
    for (unsigned i = 0; i < uncontrDimsA.size(); ++i) {
      reqDimsA.push_back(getAffineDimExpr(dimsPosC[i], ctx));
    }
    for (auto pos : contractionDims) {
      reqDimsA.push_back(getAffineDimExpr(pos, ctx));
    }
    auto transposeMap =
        AffineMap::get(mapA.getNumInputs(), 0, ArrayRef<AffineExpr>(reqDimsA));

    Value transposedA = transposeMemRef(op, rewriter, matA, mapA, transposeMap);

    rewriter.create<cim::GemmOp>(op->getLoc(), tileId, transposedA, matC);
  } else {
    rewriter.create<cim::GemmOp>(op->getLoc(), tileId, matA, matC);
  }
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
  for (auto dim : dimsA) {
    reqDimsB.push_back(dim);
  }
  for (auto dim : dimsC) {
    reqDimsB.push_back(dim);
  }
  auto transposeMap = AffineMap::get(2, 0, ArrayRef<AffineExpr>(reqDimsB));

  // Check if B needs to be transposed
  if (mapB != transposeMap) {
    Value transposedB = transposeMemRef(op, rewriter, matB, mapB, transposeMap);

    rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), tileId, transposedB);
  } else {
    rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), tileId, matB);
  }

  rewriter.create<cim::GevmOp>(op->getLoc(), tileId, matA, matC);
}

static void replaceOpWithCIMMatmul(Operation *op, PatternRewriter &rewriter) {
  auto matA = op->getOperand(0);
  auto matB = op->getOperand(1);
  auto matC = op->getOperand(2);

  auto cimTileID = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));

  // TODO(adam-smnk) Check if MatmulOp can be replaced by vector-matrix mul
  if (isa<linalg::MatmulOp>(op)) {
    // Assumes that linalg.matmul always has correct values and memrefs
    rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), cimTileID, matB);
    rewriter.create<cim::GemmOp>(op->getLoc(), cimTileID, matA, matC);
  } else if (isGevm(cast<linalg::GenericOp>(op))) {
    createCIMGevmOp(op, rewriter, cimTileID);
  } else {
    createCIMGemmOp(op, rewriter, cimTileID);
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
    target.addLegalOp<AllocOp>();
    target.addLegalOp<DeallocOp>();
    target.addLegalOp<DimOp>();
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
