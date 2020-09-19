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

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

static llvm::cl::opt<unsigned> clNumTiles(
    "cim-num-tiles",
    llvm::cl::desc(
        "Number of CIM device tiles which can be used for parallelization"),
    llvm::cl::Optional, llvm::cl::MiscFlags::DefaultOption,
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<unsigned> clTileSize(
    "cim-tile-size",
    llvm::cl::desc("CIM device tile sizes by which to tile cim operations"),
    llvm::cl::Optional, llvm::cl::MiscFlags::DefaultOption,
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool>
    clMinimizeWrites("cim-min-writes",
                     llvm::cl::desc("Performs additional transformations to "
                                    "minimize writes to CIM crossbar"),
                     llvm::cl::Optional, llvm::cl::MiscFlags::DefaultOption,
                     llvm::cl::cat(clOptionsCategory));

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

  auto contrDims = contractionReductionDims<unsigned>(dimsA, dimsB);
  auto outputDims = contractionOutputDims<unsigned>(dimsA, dimsB);

  return contrDims.size() > 0 && outputDims == dimsC;
}

static void populateTiledGEMMLoops(Operation *op, PatternRewriter &rewriter,
                                   ConstantOp &tileId, const Value &matA,
                                   const Value &matB, const Value &matC,
                                   const Value &sizeTile, bool minWrites,
                                   SmallVectorImpl<loop::ForOp> &loops,
                                   SmallVectorImpl<Value> &loopIterators) {
  // Set insertion point before the loops
  rewriter.setInsertionPoint(loops.front());

  auto dimsC = getMemRefSizes(op, rewriter, matC);
  bool isMatrix = dimsC.size() == 2;
  unsigned numCimTiles = clNumTiles == 0 ? 1 : clNumTiles;

  SmallVector<Value, 8U> cimTileIds;
  for (unsigned i = 0; i < numCimTiles; ++i) {
    cimTileIds.push_back(createIndexConst(op, rewriter, i));
  }

  Value iterM;
  Value iterN;
  Value iterK;
  if (minWrites) {
    iterM = loopIterators[2];
    iterN = loopIterators[1];
    iterK = loopIterators[0];
  } else {
    iterM = loopIterators[0];
    iterN = loopIterators[1];
    iterK = loopIterators[2];
  }

  // Populate the middle loop
  SmallVector<Value, 8U> tileB;
  Value tileC;
  SmallVector<Value, 8U> partRes;

  rewriter.setInsertionPointToStart(loops[1].getBody());
  if (minWrites) {
    for (unsigned i = 0; i < numCimTiles; ++i) {
      Value cimTileIter = createIndexConst(op, rewriter, i);
      Value cimTileIterK = rewriter.create<AddIOp>(
          op->getLoc(), rewriter.getIndexType(), iterK, cimTileIter);
      tileB.push_back(allocateTile(op, rewriter, matB, cimTileIterK, iterN,
                                   sizeTile, true));
      rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), cimTileIds[i],
                                              tileB[i]);
    }

    // Move after the innermost loop
    rewriter.setInsertionPoint(loops[1].getBody(),
                               --(loops[1].getBody()->end()));
    for (unsigned i = 0; i < numCimTiles; ++i) {
      rewriter.create<DeallocOp>(op->getLoc(), tileB[i]);
    }
  } else {
    tileC = allocateTile(op, rewriter, matC, iterM, iterN, sizeTile, true);

    for (unsigned i = 0; i < numCimTiles; ++i) {
      partRes.push_back(
          allocateTile(op, rewriter, matC, iterM, iterN, sizeTile, false));
    }

    // Move after innermost loop
    rewriter.setInsertionPoint(loops[1].getBody(),
                               --(loops[1].getBody()->end()));
    storeTile(op, rewriter, tileC, matC, iterM, iterN, sizeTile);

    rewriter.create<DeallocOp>(op->getLoc(), tileC);
    for (unsigned i = 0; i < numCimTiles; ++i) {
      rewriter.create<DeallocOp>(op->getLoc(), partRes[i]);
    }
  }

  // Populate the innermost loop
  SmallVector<Value, 8U> tileA;
  rewriter.setInsertionPointToStart(loops[2].getBody());
  for (unsigned i = 0; i < numCimTiles; ++i) {
    Value cimTileIter = createIndexConst(op, rewriter, i);
    Value cimTileIterK = rewriter.create<AddIOp>(
        op->getLoc(), rewriter.getIndexType(), iterK, cimTileIter);
    tileA.push_back(
        allocateTile(op, rewriter, matA, iterM, cimTileIterK, sizeTile, true));
  }

  if (minWrites) {
    for (unsigned i = 0; i < numCimTiles; ++i) {
      tileC = allocateTile(op, rewriter, matC, iterM, iterN, sizeTile, true);
      partRes.push_back(
          allocateTile(op, rewriter, matC, iterM, iterN, sizeTile, false));
    }

    for (unsigned i = 0; i < numCimTiles; ++i) {
      if (isMatrix) {
        rewriter.create<cim::GemmOp>(op->getLoc(), cimTileIds[i], tileA[i],
                                     partRes[i]);
      } else {
        rewriter.create<cim::GevmOp>(op->getLoc(), cimTileIds[i], tileA[i],
                                     partRes[i]);
      }
    }

    for (unsigned i = 0; i < numCimTiles; ++i) {
      rewriter.create<cim::BarrierOp>(op->getLoc(), cimTileIds[i]);
      elementwiseAddition(op, rewriter, partRes[i], tileC);
      rewriter.create<DeallocOp>(op->getLoc(), partRes[i]);
    }

    storeTile(op, rewriter, tileC, matC, iterM, iterN, sizeTile);

    rewriter.create<DeallocOp>(op->getLoc(), tileC);
  } else {
    for (unsigned i = 0; i < numCimTiles; ++i) {
      Value cimTileIter = createIndexConst(op, rewriter, i);
      Value cimTileIterK = rewriter.create<AddIOp>(
          op->getLoc(), rewriter.getIndexType(), iterK, cimTileIter);
      tileB.push_back(allocateTile(op, rewriter, matB, cimTileIterK, iterN,
                                   sizeTile, true));
      rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), cimTileIds[i],
                                              tileB[i]);
    }

    for (unsigned i = 0; i < numCimTiles; ++i) {
      if (isMatrix) {
        rewriter.create<cim::GemmOp>(op->getLoc(), cimTileIds[i], tileA[i],
                                     partRes[i]);
      } else {
        rewriter.create<cim::GevmOp>(op->getLoc(), cimTileIds[i], tileA[i],
                                     partRes[i]);
      }
    }

    for (unsigned i = 0; i < numCimTiles; ++i) {
      rewriter.create<cim::BarrierOp>(op->getLoc(), cimTileIds[i]);
      elementwiseAddition(op, rewriter, partRes[i], tileC);
      rewriter.create<DeallocOp>(op->getLoc(), tileB[i]);
    }
  }

  for (unsigned i = 0; i < numCimTiles; ++i) {
    rewriter.create<DeallocOp>(op->getLoc(), tileA[i]);
  }

  // Set insertion point back at main body outside of the loops
  rewriter.setInsertionPointAfter(loops.front());
}

static void createCIMTiledGEMM(Operation *op, PatternRewriter &rewriter,
                               ConstantOp &tileId, const Value &matA,
                               const Value &matB, const Value &matC,
                               uint32_t tileSize, bool minWrites) {
  auto dimsA = getMemRefSizes(op, rewriter, matA);
  auto dimsC = getMemRefSizes(op, rewriter, matC);

  bool isMatrix = dimsC.size() == 2;
  unsigned numCimTiles = clNumTiles == 0 ? 1 : clNumTiles;

  SmallVector<Value, 8U> cimTileIds;
  for (unsigned i = 0; i < numCimTiles; ++i) {
    cimTileIds.push_back(createIndexConst(op, rewriter, i));
  }

  Value dimM;
  Value dimN;
  Value dimK;
  if (isMatrix) {
    dimM = dimsC[0];
    dimN = dimsC[1];
    dimK = dimsA[1];
  } else {
    dimM = createIndexConst(op, rewriter, 1);
    dimN = dimsC[0];
    dimK = dimsA[0];
  }

  Value sizeTile = createIndexConst(op, rewriter, tileSize);

  Value tiledRows = calculateNumTiles(op, rewriter, sizeTile, dimM);
  Value tiledCols = calculateNumTiles(op, rewriter, sizeTile, dimN);
  Value numTiles = calculateNumTiles(op, rewriter, sizeTile, dimK);

  Value lowerBound = createIndexConst(op, rewriter, 0);

  SmallVector<Value, 8U> steps;
  Value one = createIndexConst(op, rewriter, 1);
  if (numCimTiles > 1) {
    Value cimTilesCount = createIndexConst(op, rewriter, numCimTiles);
    if (minWrites) {
      steps = {cimTilesCount, one, one};
    } else {
      steps = {one, one, cimTilesCount};
    }
  } else {
    steps = {one, one, one};
  }

  // C[M][N] = A[M][K] * B[K][N]
  SmallVector<Value, 8U> upperBounds;
  if (minWrites) {
    // Iterators: k, n, m
    upperBounds = {numTiles, tiledCols, tiledRows};
  } else {
    // Iterators: m, n, k
    upperBounds = {tiledRows, tiledCols, numTiles};
  }

  SmallVector<loop::ForOp, 8U> loops;
  SmallVector<Value, 8U> loopIterators;
  for (unsigned i = 0; i < upperBounds.size(); ++i) {
    auto loop = rewriter.create<loop::ForOp>(op->getLoc(), lowerBound,
                                             upperBounds[i], steps[i]);
    loops.push_back(loop);
    loopIterators.push_back(loop.getInductionVar());

    // Set insertion point inside the loop
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  populateTiledGEMMLoops(op, rewriter, tileId, matA, matB, matC, sizeTile,
                         minWrites, loops, loopIterators);
}

static void createCIMContractionOp(Operation *op, PatternRewriter &rewriter,
                                   ConstantOp &tileId, uint32_t tileSize,
                                   bool minWrites) {
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

  auto contractionDims = contractionReductionDims<unsigned>(dimsSetA, dimsSetB);
  auto uncontrDimsA = setDifference<unsigned>(dimsSetA, contractionDims);
  auto uncontrDimsB = setDifference<unsigned>(dimsSetB, contractionDims);

  SmallVector<AffineExpr, 8U> reqLeftDimsA;
  for (const auto &pos : uncontrDimsA) {
    reqLeftDimsA.push_back(getAffineDimExpr(pos, ctx));
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
  for (const auto &pos : uncontrDimsB) {
    reqRightDimsB.push_back(getAffineDimExpr(pos, ctx));
  }
  auto leftDimsB =
      AffineMapAttr::get(AffineMap::get(mapB.getNumInputs(), 0, reqLeftDimsB));
  auto rightDimsB =
      AffineMapAttr::get(AffineMap::get(mapB.getNumInputs(), 0, reqRightDimsB));

  // Flatten tensor to matrix
  Value flatB = groupDimensions(op, rewriter, matB, mapB,
                                ArrayAttr::get({leftDimsB, rightDimsB}, ctx));

  ArrayAttr dimsFlatC = ArrayAttr::get({leftDimsA, rightDimsB}, ctx);

  // Flatten tensor to matrix
  Value flatC = groupDimensions(op, rewriter, matC, mapC, dimsFlatC);

  if (tileSize > 0) {
    createCIMTiledGEMM(op, rewriter, tileId, flatA, flatB, flatC, tileSize,
                       minWrites);
  } else {
    rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), tileId, flatB);
    rewriter.create<cim::GemmOp>(op->getLoc(), tileId, flatA, flatC);
    rewriter.create<cim::BarrierOp>(op->getLoc(), tileId);
  }

  AffineMap mapFlatC = combineMaps(dimsFlatC);
  SmallVector<AffineExpr, 8U> outputPermutation =
      getPermutation(mapFlatC, mapC, ctx);
  auto outputPermutationPos = getDimsPositions(outputPermutation);

  // Unflatten the contraction result and copy to the output tensor
  reshapeCopy(op, rewriter, flatC, matC, dimsFlatC, outputPermutationPos, true);
}

static void createCIMGemmOp(Operation *op, PatternRewriter &rewriter,
                            ConstantOp &tileId, uint32_t tileSize,
                            bool minWrites) {
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

  auto contractionDims = contractionReductionDims<unsigned>(dimsSetA, dimsSetB);
  auto uncontrDimsA = setDifference<unsigned>(dimsSetA, contractionDims);
  auto uncontrDimsB = setDifference<unsigned>(dimsSetB, contractionDims);

  auto transposeReqs = checkGEMMTransposes(dimsPosA, dimsPosB, dimsPosC);

  Value inputB = matB;

  if (transposeReqs.transposeB) {
    SmallVector<AffineExpr, 8U> reqDimsB;
    for (const auto &pos : contractionDims) {
      reqDimsB.push_back(getAffineDimExpr(pos, ctx));
    }
    for (const auto &pos : uncontrDimsB) {
      reqDimsB.push_back(getAffineDimExpr(pos, ctx));
    }
    auto transposeMap =
        AffineMap::get(mapB.getNumInputs(), 0, ArrayRef<AffineExpr>(reqDimsB));

    inputB = transposeMemRef(op, rewriter, matB, mapB, transposeMap);
  }

  Value inputA = matA;
  Value outputC = allocateDuplicate(op, rewriter, matC);

  if (transposeReqs.transposeA) {
    SmallVector<AffineExpr, 8U> reqDimsA;
    for (const auto &pos : uncontrDimsA) {
      reqDimsA.push_back(getAffineDimExpr(pos, ctx));
    }
    for (const auto &pos : contractionDims) {
      reqDimsA.push_back(getAffineDimExpr(pos, ctx));
    }
    auto transposeMap =
        AffineMap::get(mapA.getNumInputs(), 0, ArrayRef<AffineExpr>(reqDimsA));

    inputA = transposeMemRef(op, rewriter, matA, mapA, transposeMap);
  }

  if (tileSize > 0) {
    Value zero = rewriter.create<ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(
            outputC.getType().cast<MemRefType>().getElementType(), 0));
    rewriter.create<FillOp>(op->getLoc(), outputC, zero);

    createCIMTiledGEMM(op, rewriter, tileId, inputA, inputB, outputC, tileSize,
                       minWrites);
  } else {
    rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), tileId, inputB);
    rewriter.create<cim::GemmOp>(op->getLoc(), tileId, inputA, outputC);
    rewriter.create<cim::BarrierOp>(op->getLoc(), tileId);
  }

  if (transposeReqs.transposeResult) {
    SmallVector<AffineExpr, 8U> reqLeftDimsOutput;
    for (const auto &pos : uncontrDimsA) {
      reqLeftDimsOutput.push_back(getAffineDimExpr(pos, ctx));
    }
    SmallVector<AffineExpr, 8U> reqRightDimsOutput;
    for (const auto &pos : uncontrDimsB) {
      reqRightDimsOutput.push_back(getAffineDimExpr(pos, ctx));
    }

    auto leftDimsOutput = AffineMapAttr::get(
        AffineMap::get(mapC.getNumInputs(), 0, reqLeftDimsOutput));
    auto rightDimsOutput = AffineMapAttr::get(
        AffineMap::get(mapC.getNumInputs(), 0, reqRightDimsOutput));

    ArrayAttr outputDims =
        ArrayAttr::get({leftDimsOutput, rightDimsOutput}, ctx);
    AffineMap outputCMap = combineMaps(outputDims);

    SmallVector<AffineExpr, 8U> outputPermutation =
        getPermutation(outputCMap, mapC, ctx);
    auto outputPermutationPos = getDimsPositions(outputPermutation);

    // transpose GEMM results then add it to the output matrix
    reshapeCopy(op, rewriter, outputC, matC, outputDims, outputPermutationPos,
                true);
  } else {
    elementwiseAddition(op, rewriter, outputC, matC);
  }
}

static void createCIMGevmOp(Operation *op, PatternRewriter &rewriter,
                            ConstantOp &tileId, uint32_t tileSize,
                            bool minWrites) {
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

  // Check if B needs to be transposed
  if (mapB != transposeMap) {
    inputB = transposeMemRef(op, rewriter, matB, mapB, transposeMap);
  }

  if (tileSize > 0) {
    createCIMTiledGEMM(op, rewriter, tileId, matA, inputB, matC, tileSize,
                       minWrites);
  } else {
    Value outputC = allocateDuplicate(op, rewriter, matC);

    rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), tileId, inputB);
    rewriter.create<cim::GevmOp>(op->getLoc(), tileId, matA, outputC);
    rewriter.create<cim::BarrierOp>(op->getLoc(), tileId);

    elementwiseAddition(op, rewriter, outputC, matC);
  }
}

static void createCIMGemvOp(Operation *op, PatternRewriter &rewriter,
                            ConstantOp &tileId, uint32_t tileSize,
                            bool minWrites) {
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

  createCIMGevmOp(op, rewriter, tileId, tileSize, minWrites);
}

// TODO(adam-smnk) Make lowering more progressive, move some of the
// conversions into CIMtoSTD pass
static void replaceOpWithCIMMatmul(Operation *op, PatternRewriter &rewriter,
                                   uint32_t tileSize, bool minWrites) {
  auto matA = op->getOperand(0);
  auto matB = op->getOperand(1);
  auto matC = op->getOperand(2);

  auto cimTileID = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));

  // TODO(adam-smnk) Check if MatmulOp/GEMM can be replaced by vector-matrix
  // mul
  if (isa<linalg::MatmulOp>(op)) {
    // Assumes that linalg.matmul always has correct values and memrefs
    rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), cimTileID, matB);
    rewriter.create<cim::GemmOp>(op->getLoc(), cimTileID, matA, matC);
    rewriter.create<cim::BarrierOp>(op->getLoc(), cimTileID);
  } else if (isGevm(cast<linalg::GenericOp>(op))) {
    createCIMGevmOp(op, rewriter, cimTileID, tileSize, minWrites);
  } else if (isGemm(cast<linalg::GenericOp>(op))) {
    createCIMGemmOp(op, rewriter, cimTileID, tileSize, minWrites);
  } else if (isGemv(cast<linalg::GenericOp>(op))) {
    createCIMGemvOp(op, rewriter, cimTileID, tileSize, minWrites);
  } else {
    createCIMContractionOp(op, rewriter, cimTileID, tileSize, minWrites);
  }

  rewriter.eraseOp(op);
}

// TODO(adam-smnk) Add support for matmul with accumulation
struct MatmulOpLowering : public OpRewritePattern<linalg::MatmulOp> {
  // using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  MatmulOpLowering() = delete;
  MatmulOpLowering(MLIRContext *ctx, uint32_t tileSize_, bool minWrites_)
      : OpRewritePattern(ctx), tileSize(tileSize_), minWrites(minWrites_) {}

  PatternMatchResult matchAndRewrite(linalg::MatmulOp op,
                                     PatternRewriter &rewriter) const final {
    replaceOpWithCIMMatmul(op, rewriter, tileSize, minWrites);
    return matchSuccess();
  }

  uint32_t tileSize;
  bool minWrites;
};

// TODO(adam-smnk) Check for contiguous memory
struct GenericOpLowering : public OpRewritePattern<linalg::GenericOp> {
  // using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  GenericOpLowering() = delete;
  GenericOpLowering(MLIRContext *ctx, uint32_t tileSize_, bool minWrites_)
      : OpRewritePattern(ctx), tileSize(tileSize_), minWrites(minWrites_) {}

  PatternMatchResult matchAndRewrite(linalg::GenericOp op,
                                     PatternRewriter &rewriter) const final {
    replaceOpWithCIMMatmul(op, rewriter, tileSize, minWrites);
    return matchSuccess();
  }

  uint32_t tileSize;
  bool minWrites;
};

/// A pass that replaces Linalg operations with their corresponding CIM
/// equivalent.
class LowerLinalgOpsToCIMOpsPass
    : public OperationPass<LowerLinalgOpsToCIMOpsPass, ModuleOp> {
public:
  LowerLinalgOpsToCIMOpsPass() { LowerLinalgOpsToCIMOpsPass(0, false); }

  LowerLinalgOpsToCIMOpsPass(uint32_t tileSize_, bool minWrites_)
      : tileSize(tileSize_), minWrites(minWrites_) {}

  void runOnOperation() override {
    ModuleOp m = getOperation();

    OwningRewritePatternList patterns;
    populateLinalgToCIMConversionPatterns(patterns, &getContext(), tileSize,
                                          minWrites);

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

  uint32_t tileSize;
  bool minWrites;
};
} // anonymous namespace

void mlir::populateLinalgToCIMConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx, uint32_t tileSize,
    bool minWrites) {
  patterns.insert<MatmulOpLowering, GenericOpLowering>(ctx, tileSize,
                                                       minWrites);
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::createConvertLinalgToCIMPass(uint32_t tileSize, bool minWrites) {
  return std::make_unique<LowerLinalgOpsToCIMOpsPass>(tileSize, minWrites);
}

static PassRegistration<LowerLinalgOpsToCIMOpsPass>
    pass("convert-linalg-to-cim",
         "Convert the operations from the linalg dialect into the CIM dialect",
         [] {
           return std::make_unique<LowerLinalgOpsToCIMOpsPass>(
               clTileSize, clMinimizeWrites);
         });
