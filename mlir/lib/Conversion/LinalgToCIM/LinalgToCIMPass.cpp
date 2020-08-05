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

static auto getResultMaps(ArrayAttr affineMaps) {
  return functional::map([](AffineMapAttr a) { return a.getValue(); },
                         affineMaps.getAsRange<AffineMapAttr>());
}

static auto getResultMaps(linalg::GenericOp genericOp) {
  return getResultMaps(genericOp.indexing_maps());
}

static auto getResultDims(ArrayAttr affineMaps) {
  return functional::map(
      [](AffineMapAttr a) { return a.getValue().getResults(); },
      affineMaps.getAsRange<AffineMapAttr>());
}

static auto getResultDims(linalg::GenericOp genericOp) {
  return getResultDims(genericOp.indexing_maps());
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

static std::vector<unsigned>
getDimsPositions(const ArrayRef<AffineExpr> &affineDims) {
  std::vector<unsigned> dims;

  for (const auto &dim : affineDims) {
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
    for (const auto &dimPos : reqDimsPos) {
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

static SmallVector<Value, 8U>
getMemRefSizes(Operation *op, PatternRewriter &rewriter, const Value &memRef,
               const ArrayRef<unsigned> &dimsPositions) {
  SmallVector<Value, 8U> dimOperands;

  for (const auto &pos : dimsPositions) {
    dimOperands.push_back(rewriter.create<DimOp>(op->getLoc(), memRef, pos));
  }

  return dimOperands;
}

static Value calculateDimsSize(Operation *op, PatternRewriter &rewriter,
                               const ArrayRef<Value> &dimOperands) {
  Value dimsSize = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

  for (const auto &dim : dimOperands) {
    dimsSize = rewriter.create<MulIOp>(op->getLoc(), rewriter.getIndexType(),
                                       dimsSize, dim);
  }

  return dimsSize;
}

static SmallVector<Value, 8U>
calculateDimsStrides(Operation *op, PatternRewriter &rewriter,
                     const ArrayRef<Value> &dimOperands) {
  SmallVector<Value, 8U> strides;

  Value one = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
  strides.push_back(one);

  for (auto it = dimOperands.rbegin(); it != (dimOperands.rend() - 1); ++it) {
    strides.push_back(rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(), strides.back(), *it));
  }
  std::reverse(strides.begin(), strides.end());

  return strides;
}

// Expects the strides to match the iterators, e.g.
// array: arr[I][J][K]; iterators: {i,j,k}; strides: {i*J*K, j*K, 1}
static Value calculateLinearIndex(Operation *op, PatternRewriter &rewriter,
                                  const ArrayRef<Value> &dimIterators,
                                  const ArrayRef<Value> &dimStrides) {
  assert(dimIterators.size() == dimStrides.size() &&
         "Number of iterators and strides differ");

  Value linearIndex = dimIterators.back();

  for (unsigned i = 0; i < dimIterators.size() - 1; ++i) {
    Value mulRes = rewriter.create<MulIOp>(
        op->getLoc(), rewriter.getIndexType(), dimIterators[i], dimStrides[i]);
    linearIndex = rewriter.create<AddIOp>(op->getLoc(), rewriter.getIndexType(),
                                          linearIndex, mulRes);
  }

  return linearIndex;
}

static SmallVector<AffineExpr, 8U> getPermutation(const AffineMap &originalMap,
                                                  const AffineMap &targetMap,
                                                  MLIRContext *context) {
  auto dimsPos = getDimsPositions(originalMap.getResults());
  auto targetDimsPos = getDimsPositions(targetMap.getResults());

  SmallVector<AffineExpr, 8U> permutation;
  for (unsigned i = 0; i < targetDimsPos.size(); ++i) {
    auto it = std::find(dimsPos.begin(), dimsPos.end(), targetDimsPos[i]);
    int pos = std::distance(dimsPos.begin(), it);
    permutation.push_back(getAffineDimExpr(pos, context));
  }

  return permutation;
}

// Performs reshape with copy.
// Reassociation defines continuous dimension groupings for the lower rank
// memref.
// Expects both memrefs to point to contiguous memory.
// Permutation list should reorder all the dimensions present in the higher rank
// memref or be empty, otherwise the behavior is undefined.
static void reshapeCopy(Operation *op, PatternRewriter &rewriter,
                        const Value &inputMemRef, const Value &outputMemRef,
                        const ArrayAttr &reassociation,
                        ArrayRef<unsigned> permutation = ArrayRef<unsigned>(),
                        bool performElementwiseSum = false) {
  Value memRef = inputMemRef;

  // In case of dimension expansion, treat reassociation as inverse maps
  unsigned inputNumDims =
      inputMemRef.getType().cast<MemRefType>().getShape().size();
  unsigned outputNumDims =
      outputMemRef.getType().cast<MemRefType>().getShape().size();
  bool isInverseMap = inputNumDims < outputNumDims;

  // All loops and strides are calculated based on the memref
  // with higher rank
  if (isInverseMap) {
    memRef = outputMemRef;
  }

  SmallVector<AffineMap, 8U> targetMaps = getResultMaps(reassociation);

  // Determine desired order of the dimensions of the higher rank memref.
  // In case of no permutation, use the default dimension order.
  SmallVector<unsigned, 8U> positions;
  if (permutation.empty()) {
    unsigned numDims = memRef.getType().cast<MemRefType>().getShape().size();
    for (unsigned i = 0; i < numDims; ++i) {
      positions.push_back(i);
    }

    permutation = positions;
  }

  SmallVector<Value, 8U> dimOperands =
      getMemRefSizes(op, rewriter, memRef, permutation);

  // For each grouping, gather corresponding dimension sizes
  SmallVector<SmallVector<Value, 8U>, 8U> targetDimOperands;
  auto itBegin = dimOperands.begin();
  for (const auto &map : targetMaps) {
    auto itEnd = itBegin + map.getNumResults();
    targetDimOperands.push_back(SmallVector<Value, 8U>(itBegin, itEnd));
    itBegin = itEnd;
  }

  // For each grouping, compute their strides at runtime
  SmallVector<SmallVector<Value, 8U>, 8U> targetDimStrides;
  for (unsigned i = 0; i < targetDimOperands.size(); ++i) {
    targetDimStrides.push_back(
        calculateDimsStrides(op, rewriter, targetDimOperands[i]));
  }

  Value lowerBound = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  Value step = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

  // Loop over higher rank memref dimensions
  SmallVector<loop::ForOp, 8U> loops;
  SmallVector<Value, 8U> loopIterators;
  for (unsigned i = 0; i < dimOperands.size(); ++i) {
    // Iterate in the original dimension order before permutation
    auto it = std::find(permutation.begin(), permutation.end(), i);
    int pos = std::distance(permutation.begin(), it);
    Value upperBound = dimOperands[pos];

    auto loop = rewriter.create<loop::ForOp>(op->getLoc(), lowerBound,
                                             upperBound, step);
    loops.push_back(loop);
    loopIterators.push_back(loop.getInductionVar());

    // Set insertion point inside the loop
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  // Perform copy in the inner-most loop
  SmallVector<SmallVector<Value, 8U>, 8U> targetDimIters;
  unsigned start = 0;
  for (unsigned i = 0; i < targetDimOperands.size(); ++i) {
    unsigned end = start + targetDimOperands[i].size();
    SmallVector<Value, 8U> iters;
    for (unsigned j = start; j < end; ++j) {
      iters.push_back(loopIterators[permutation[j]]);
    }
    targetDimIters.push_back(iters);
    start = end;
  }

  // For each grouping, compute their linear indices at runtime
  SmallVector<Value, 8U> indices;
  for (unsigned i = 0; i < targetDimIters.size(); ++i) {
    indices.push_back(calculateLinearIndex(op, rewriter, targetDimIters[i],
                                           targetDimStrides[i]));
  }

  ValueRange inputIndices = ValueRange(loopIterators);
  ValueRange outputIndices = ValueRange(indices);

  if (isInverseMap) {
    inputIndices = ValueRange(indices);
    outputIndices = ValueRange(loopIterators);
  }

  Value inputElement =
      rewriter.create<LoadOp>(op->getLoc(), inputMemRef, inputIndices);

  // TODO(adam-smnk) Refactor bool flag into more modular solution.
  if (performElementwiseSum) {
    Value outputElement =
        rewriter.create<LoadOp>(op->getLoc(), outputMemRef, outputIndices);
    inputElement = rewriter.create<AddIOp>(
        op->getLoc(), inputMemRef.getType().cast<MemRefType>().getElementType(),
        inputElement, outputElement);
  }

  rewriter.create<StoreOp>(op->getLoc(), inputElement, outputMemRef,
                           outputIndices);

  // Set insertion point back at main body outside of the loops
  rewriter.setInsertionPoint(op);
}

static Value allocateDuplicate(Operation *op, PatternRewriter &rewriter,
                               const Value &memRef) {
  const auto memRefType = memRef.getType().cast<MemRefType>();

  SmallVector<unsigned, 8U> dimPositions;
  SmallVector<int64_t, 8U> duplicateSizes;

  const unsigned numDims = memRefType.getShape().size();
  for (unsigned i = 0; i < numDims; ++i) {
    dimPositions.push_back(i);
    duplicateSizes.push_back(-1);
  }

  // Get input memref sizes at runtime
  SmallVector<Value, 8U> dimOperands =
      getMemRefSizes(op, rewriter, memRef, dimPositions);

  MemRefType duplicateMemRefType =
      MemRefType::Builder(duplicateSizes, memRefType.getElementType());

  Value duplicateMemRef =
      rewriter.create<AllocOp>(op->getLoc(), duplicateMemRefType, dimOperands)
          .getResult();

  // Deallocate the data at the end of block after CIM computations are
  // finished
  auto deallocOp =
      rewriter.create<DeallocOp>(op->getLoc(), duplicateMemRef).getOperation();
  deallocOp->moveBefore(&(deallocOp->getBlock()->back()));

  return duplicateMemRef;
}

static void elementwiseAddition(Operation *op, PatternRewriter &rewriter,
                                const Value &inputMemRef,
                                const Value &outputMemRef) {
  assert(inputMemRef.getType().cast<MemRefType>().getShape() ==
             outputMemRef.getType().cast<MemRefType>().getShape() &&
         "Input and output shapes differ");

  const auto memRefType = inputMemRef.getType().cast<MemRefType>();
  const unsigned numDims = memRefType.getShape().size();

  SmallVector<unsigned, 8U> dimPositions;
  for (unsigned i = 0; i < numDims; ++i) {
    dimPositions.push_back(i);
  }

  // Get memref sizes at runtime
  SmallVector<Value, 8U> dimOperands =
      getMemRefSizes(op, rewriter, inputMemRef, dimPositions);

  Value lowerBound = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  Value step = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

  // Loop over higher rank memref dimensions
  SmallVector<loop::ForOp, 8U> loops;
  SmallVector<Value, 8U> loopIterators;
  for (unsigned i = 0; i < numDims; ++i) {
    Value upperBound = dimOperands[i];

    auto loop = rewriter.create<loop::ForOp>(op->getLoc(), lowerBound,
                                             upperBound, step);
    loops.push_back(loop);
    loopIterators.push_back(loop.getInductionVar());

    // Set insertion point inside the loop
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  Value inputElement = rewriter.create<LoadOp>(op->getLoc(), inputMemRef,
                                               ValueRange(loopIterators));
  Value outputElement = rewriter.create<LoadOp>(op->getLoc(), outputMemRef,
                                                ValueRange(loopIterators));
  Value sumResult = rewriter.create<AddIOp>(
      op->getLoc(), memRefType.getElementType(), inputElement, outputElement);
  rewriter.create<StoreOp>(op->getLoc(), sumResult, outputMemRef,
                           ValueRange(loopIterators));

  // Set insertion point back at main body outside of the loops
  rewriter.setInsertionPoint(op);
}

// Groups (collapses) dimensions of the input memref based on a reassociation
// map. Allows for permutation.
// Performs alloc and copy of the original data.
static Value groupDimensions(Operation *op, PatternRewriter &rewriter,
                             const Value &memRef, const AffineMap &memRefMap,
                             const ArrayAttr &reassociation) {
  auto *ctx = op->getContext();

  SmallVector<AffineMap, 8U> targetMaps = getResultMaps(reassociation);

  SmallVector<AffineExpr, 8U> targetDims;
  for (const auto &map : targetMaps) {
    auto dims = map.getResults();
    targetDims.insert(targetDims.end(), dims.begin(), dims.end());
  }
  SmallVector<AffineExpr, 8U> outputPermutation = getPermutation(
      memRefMap, AffineMap::get(targetDims.size(), 0, targetDims), ctx);

  auto outputPermutationPos = getDimsPositions(outputPermutation);

  // Get input memref sizes at runtime
  SmallVector<Value, 8U> dimOperands =
      getMemRefSizes(op, rewriter, memRef, outputPermutationPos);

  // For each grouping, gather corresponding dimension sizes
  SmallVector<SmallVector<Value, 8U>, 8U> targetDimOperands;
  auto itBegin = dimOperands.begin();
  for (const auto &map : targetMaps) {
    auto itEnd = itBegin + map.getNumResults();
    targetDimOperands.push_back(SmallVector<Value, 8U>(itBegin, itEnd));
    itBegin = itEnd;
  }

  // For each grouping, calculate their sizes at runtime
  SmallVector<Value, 8U> targetDimSizes;
  for (const auto &operands : targetDimOperands) {
    targetDimSizes.push_back(calculateDimsSize(op, rewriter, operands));
  }
  SmallVector<int64_t, 8U> targetMemRefSizes;
  for (unsigned i = 0; i < reassociation.size(); ++i) {
    targetMemRefSizes.push_back(-1);
  }

  MemRefType outputMemRefType = MemRefType::Builder(
      targetMemRefSizes, memRef.getType().cast<MemRefType>().getElementType());

  Value outputMemRef =
      rewriter.create<AllocOp>(op->getLoc(), outputMemRefType, targetDimSizes)
          .getResult();

  // Deallocate the data at the end of block after CIM computations are
  // finished
  auto deallocOp =
      rewriter.create<DeallocOp>(op->getLoc(), outputMemRef).getOperation();
  deallocOp->moveBefore(&(deallocOp->getBlock()->back()));

  reshapeCopy(op, rewriter, memRef, outputMemRef, reassociation,
              outputPermutationPos);

  return outputMemRef;
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
