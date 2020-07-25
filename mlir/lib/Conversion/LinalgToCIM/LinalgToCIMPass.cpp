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

  auto contrDimsA = setDifference<unsigned>(dimsA, uncontrDimsA);
  auto contrDimsB = setDifference<unsigned>(dimsB, uncontrDimsB);
  auto contrDims = setUnion<unsigned>(contrDimsA, contrDimsB);

  auto outputDims = setUnion<unsigned>(uncontrDimsA, uncontrDimsB);

  return contrDims.size() > 0 && contrDimsA == contrDimsB &&
         dimsC.size() == (uncontrDimsA.size() + uncontrDimsB.size()) &&
         outputDims == dimsC;
}

struct TransposeAnalysisResults {
  bool transposeA;
  bool transposeB;

  TransposeAnalysisResults() : TransposeAnalysisResults(false, false){};
  TransposeAnalysisResults(bool transposeA_, bool transposeB_)
      : transposeA(transposeA_), transposeB(transposeB_){};
};

// TODO(adam-smnk) Split the need to transpose contracted and
// uncontracted dimensions to avoid unnecessary tranposition when
// contraction dimensions already match.
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

  // check if A and C dimensions match
  for (unsigned i = 0; i < uncontrDimsA.size(); ++i) {
    if (dimsPosA[i] != dimsPosC[i]) {
      return TransposeAnalysisResults(true, true);
    }
  }

  // check if B and C dimensions match
  for (unsigned i = 0; i < uncontrDimsB.size(); ++i) {
    unsigned bPos = contractionDims.size() + i;
    unsigned cPos = uncontrDimsA.size() + i;

    if (dimsPosB[bPos] != dimsPosC[cPos]) {
      return TransposeAnalysisResults(true, true);
    }
  }

  return TransposeAnalysisResults(false, false);
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

  // Copy only if there are any memory layout changes required
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

static void copyMatrixToTensor(Operation *op, PatternRewriter &rewriter,
                               const Value &memRef, const AffineMap &memRefMap,
                               const Value &targetMemRef,
                               const AffineMap &targetMap, uint32_t numLeftDims,
                               uint32_t numRightDims) {
  auto *ctx = cast<GenericOp>(op).getContext();

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

  MemRefType transposedMemRef =
      MemRefType::Builder({-1, -1}, memRefType.getElementType());

  auto outputPermutationPos =
      getDimsPositions(ArrayRef<AffineExpr>(outputPermutation));

  SmallVector<Value, 8U> dimOperands;
  for (unsigned i = 0; i < outputPermutationPos.size(); ++i) {
    dimOperands.push_back(
        rewriter.create<DimOp>(op->getLoc(), memRef, outputPermutationPos[i]));
  }

  Value one = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

  SmallVector<Value, 8U> leftDimStrides;
  leftDimStrides.push_back(one);
  for (unsigned i = 1; i < numLeftDims; ++i) {
    leftDimStrides.push_back(rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(), leftDimStrides.back(),
        dimOperands[numLeftDims - i]));
  }
  std::reverse(leftDimStrides.begin(), leftDimStrides.end());

  SmallVector<Value, 8U> rightDimStrides;
  rightDimStrides.push_back(one);
  for (unsigned i = 1; i < numRightDims; ++i) {
    rightDimStrides.push_back(rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(), rightDimStrides.back(),
        dimOperands[dimOperands.size() - i]));
  }
  std::reverse(rightDimStrides.begin(), rightDimStrides.end());

  Value lowerBound = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  Value step = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

  SmallVector<loop::ForOp, 8U> loops;
  for (unsigned i = 0; i < dimOperands.size(); ++i) {
    auto it =
        std::find(outputPermutationPos.begin(), outputPermutationPos.end(), i);
    int pos = std::distance(outputPermutationPos.begin(), it);
    Value upperBound = dimOperands[pos];

    auto loop = rewriter.create<loop::ForOp>(op->getLoc(), lowerBound,
                                             upperBound, step);
    loops.push_back(loop);

    // Set insertion point inside the loop
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  // Perform copy in the inner-most loop body
  SmallVector<Value, 8U> loopIterators;
  for (auto loop : loops) {
    loopIterators.push_back(loop.getInductionVar());
  }

  // Copy based on strides, loop: a, b, c, d:
  // flatA[a * stride b + b][c * stride d + d] = A[a][b][c][d]

  Value zero = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  Value rowPos = rewriter.create<AddIOp>(
      op->getLoc(), rewriter.getIndexType(), zero,
      loopIterators[outputPermutationPos[numLeftDims - 1]]);

  for (unsigned i = 0; i < numLeftDims - 1; ++i) {
    Value mulRes = rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(),
        loopIterators[outputPermutationPos[i]], leftDimStrides[i]);
    rowPos = rewriter.create<AddIOp>(op->getLoc(), rewriter.getIndexType(),
                                     rowPos, mulRes);
  }

  Value colPos =
      rewriter.create<AddIOp>(op->getLoc(), rewriter.getIndexType(), zero,
                              loopIterators[outputPermutationPos.back()]);

  for (unsigned i = 0; i < numRightDims - 1; ++i) {
    Value mulRes = rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(),
        loopIterators[outputPermutationPos[numLeftDims + i]],
        rightDimStrides[i]);
    colPos = rewriter.create<AddIOp>(op->getLoc(), rewriter.getIndexType(),
                                     colPos, mulRes);
  }

  Value element = rewriter.create<LoadOp>(
      op->getLoc(), targetMemRef, ValueRange(ArrayRef<Value>{rowPos, colPos}));
  rewriter.create<StoreOp>(op->getLoc(), element, memRef,
                           ValueRange(ArrayRef<Value>(loopIterators)));

  // Set insertion point back at main body outside of loops
  rewriter.setInsertionPoint(op);
}

static Value flattenTensorToMatrix(Operation *op, PatternRewriter &rewriter,
                                   const Value &memRef,
                                   const AffineMap &memRefMap,
                                   const AffineMap &targetMap,
                                   uint32_t numLeftDims,
                                   uint32_t numRightDims) {
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

  MemRefType transposedMemRef =
      MemRefType::Builder({-1, -1}, memRefType.getElementType());

  auto outputPermutationPos =
      getDimsPositions(ArrayRef<AffineExpr>(outputPermutation));

  SmallVector<Value, 8U> dimOperands;
  for (unsigned i = 0; i < outputPermutationPos.size(); ++i) {
    dimOperands.push_back(
        rewriter.create<DimOp>(op->getLoc(), memRef, outputPermutationPos[i]));
  }

  Value leftDimsSize = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
  for (unsigned i = 0; i < numLeftDims; ++i) {
    leftDimsSize = rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(), leftDimsSize, dimOperands[i]);
  }
  Value rightDimsSize = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
  for (unsigned i = numLeftDims; i < dimOperands.size(); ++i) {
    rightDimsSize = rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(), rightDimsSize, dimOperands[i]);
  }

  transposedMemory =
      rewriter
          .create<AllocOp>(op->getLoc(), transposedMemRef,
                           ArrayRef<Value>{leftDimsSize, rightDimsSize})
          .getResult();
  auto deallocOp =
      rewriter.create<DeallocOp>(op->getLoc(), transposedMemory).getOperation();
  deallocOp->moveBefore(&(deallocOp->getBlock()->back()));
  // rewriter.create<linalg::CopyOp>(op->getLoc(), memRef, transposedMemory,
  //                                 AffineMapAttr::get(inputPermutationMap),
  //                                 AffineMapAttr::get(outputPermutationMap));

  Value one = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

  SmallVector<Value, 8U> leftDimStrides;
  leftDimStrides.push_back(one);
  for (unsigned i = 1; i < numLeftDims; ++i) {
    leftDimStrides.push_back(rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(), leftDimStrides.back(),
        dimOperands[numLeftDims - i]));
  }
  std::reverse(leftDimStrides.begin(), leftDimStrides.end());

  SmallVector<Value, 8U> rightDimStrides;
  rightDimStrides.push_back(one);
  for (unsigned i = 1; i < numRightDims; ++i) {
    rightDimStrides.push_back(rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(), rightDimStrides.back(),
        dimOperands[dimOperands.size() - i]));
  }
  std::reverse(rightDimStrides.begin(), rightDimStrides.end());

  Value lowerBound = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  Value step = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

  SmallVector<loop::ForOp, 8U> loops;
  for (unsigned i = 0; i < dimOperands.size(); ++i) {
    auto it =
        std::find(outputPermutationPos.begin(), outputPermutationPos.end(), i);
    int pos = std::distance(outputPermutationPos.begin(), it);
    Value upperBound = dimOperands[pos];

    auto loop = rewriter.create<loop::ForOp>(op->getLoc(), lowerBound,
                                             upperBound, step);
    loops.push_back(loop);

    // Set insertion point inside the loop
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  // Perform copy in the inner-most loop body
  SmallVector<Value, 8U> loopIterators;
  for (auto loop : loops) {
    loopIterators.push_back(loop.getInductionVar());
  }

  Value element = rewriter.create<LoadOp>(
      op->getLoc(), memRef, ValueRange(ArrayRef<Value>(loopIterators)));

  // Copy based on strides, loop: a, b, c, d:
  // flatA[a * stride b + b][c * stride d + d] = A[a][b][c][d]

  Value zero = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  Value rowPos = rewriter.create<AddIOp>(
      op->getLoc(), rewriter.getIndexType(), zero,
      loopIterators[outputPermutationPos[numLeftDims - 1]]);

  for (unsigned i = 0; i < numLeftDims - 1; ++i) {
    Value mulRes = rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(),
        loopIterators[outputPermutationPos[i]], leftDimStrides[i]);
    rowPos = rewriter.create<AddIOp>(op->getLoc(), rewriter.getIndexType(),
                                     rowPos, mulRes);
  }

  Value colPos =
      rewriter.create<AddIOp>(op->getLoc(), rewriter.getIndexType(), zero,
                              loopIterators[outputPermutationPos.back()]);

  for (unsigned i = 0; i < numRightDims - 1; ++i) {
    Value mulRes = rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(),
        loopIterators[outputPermutationPos[numLeftDims + i]],
        rightDimStrides[i]);
    colPos = rewriter.create<AddIOp>(op->getLoc(), rewriter.getIndexType(),
                                     colPos, mulRes);
  }

  rewriter.create<StoreOp>(op->getLoc(), element, transposedMemory,
                           ValueRange(ArrayRef<Value>{rowPos, colPos}));

  // Set insertion point back at main body outside of loops
  rewriter.setInsertionPoint(op);

  return transposedMemory;
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

  SmallVector<AffineExpr, 8U> reqDimsA;
  for (unsigned i = 0; i < uncontrDimsA.size(); ++i) {
    reqDimsA.push_back(getAffineDimExpr(dimsPosC[i], ctx));
  }
  for (auto pos : contractionDims) {
    reqDimsA.push_back(getAffineDimExpr(pos, ctx));
  }
  auto transposeMapA =
      AffineMap::get(mapA.getNumInputs(), 0, ArrayRef<AffineExpr>(reqDimsA));

  Value flatA =
      flattenTensorToMatrix(op, rewriter, matA, mapA, transposeMapA,
                            uncontrDimsA.size(), contractionDims.size());

  SmallVector<AffineExpr, 8U> reqDimsB;
  for (auto pos : contractionDims) {
    reqDimsB.push_back(getAffineDimExpr(pos, ctx));
  }
  for (unsigned i = 0; i < uncontrDimsB.size(); ++i) {
    unsigned cPos = uncontrDimsA.size() + i;
    reqDimsB.push_back(getAffineDimExpr(dimsPosC[cPos], ctx));
  }
  auto transposeMapB =
      AffineMap::get(mapB.getNumInputs(), 0, ArrayRef<AffineExpr>(reqDimsB));

  Value flatB =
      flattenTensorToMatrix(op, rewriter, matB, mapB, transposeMapB,
                            contractionDims.size(), uncontrDimsB.size());

  Value flatC = flattenTensorToMatrix(op, rewriter, matC, mapC, mapC,
                                      uncontrDimsA.size(), uncontrDimsB.size());

  rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), tileId, flatB);
  rewriter.create<cim::GemmOp>(op->getLoc(), tileId, flatA, flatC);
  rewriter.create<cim::BarrierOp>(op->getLoc(), tileId);

  copyMatrixToTensor(op, rewriter, matC, mapC, flatC, mapC, uncontrDimsA.size(),
                     uncontrDimsB.size());
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
  } else if (isGemm(cast<linalg::GenericOp>(op))) {
    createCIMGemmOp(op, rewriter, cimTileID);
  } else {
    createCIMContractionOp(op, rewriter, cimTileID);
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
    // target.addLegalOp<ConstantOp>();
    // target.addLegalOp<AllocOp>();
    // target.addLegalOp<DeallocOp>();
    // target.addLegalOp<DimOp>();
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
