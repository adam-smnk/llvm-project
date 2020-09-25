#include "mlir/Dialect/CIM/Utils/Tiling.h"
#include "mlir/Dialect/CIM/IR/CIMDialect.h"
#include "mlir/Dialect/CIM/Utils/RuntimeUtils.h"
#include "mlir/Dialect/CIM/Utils/StaticUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::cim;

static loop::IfOp insertIfBlock(Operation *op, PatternRewriter &rewriter,
                                const Value &cond) {
  loop::IfOp ifOp = rewriter.create<loop::IfOp>(op->getLoc(), cond, false);
  rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());

  return ifOp;
}

static void exitIfBlock(Operation *op, PatternRewriter &rewriter,
                        const loop::IfOp &ifOp) {
  rewriter.setInsertionPointAfter(ifOp);
}

static loop::IfOp insertUnrollBoundaryCheck(Operation *op,
                                            PatternRewriter &rewriter,
                                            const Value &dimIter,
                                            const Value &boundaryMaxVal,
                                            unsigned tileIter) {
  Value unrollVal = createIndexConst(op, rewriter, tileIter);
  Value unrollIter = rewriter.create<AddIOp>(
      op->getLoc(), rewriter.getIndexType(), dimIter, unrollVal);
  Value cond = rewriter.create<CmpIOp>(op->getLoc(), CmpIPredicate::ult,
                                       unrollIter, boundaryMaxVal);

  return insertIfBlock(op, rewriter, cond);
}

static void populateTiledGEMMLoops(
    Operation *op, PatternRewriter &rewriter, ConstantOp &tileId,
    const Value &matA, const Value &matB, const Value &matC,
    const Value &sizeTile, bool minWrites, unsigned numTilesOption,
    SmallVectorImpl<loop::ForOp> &loops, SmallVectorImpl<Value> &loopIterators,
    bool boundaryChecks = false, ArrayRef<Value> ub = {}) {
  // Set insertion point before the loops
  rewriter.setInsertionPoint(loops.front());

  auto dimsC = getMemRefSizes(op, rewriter, matC);
  bool isMatrix = dimsC.size() == 2;
  unsigned numCimTiles = numTilesOption == 0 ? 1 : numTilesOption;

  SmallVector<Value, 8U> cimTileIds;
  for (unsigned i = 0; i < numCimTiles; ++i) {
    cimTileIds.push_back(createIntConst(op, rewriter, i, 32));
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

  Value maxK;
  if (boundaryChecks) {
    if (minWrites) {
      maxK = ub[0];
    } else {
      maxK = ub[2];
    }
  }

  // Populate the middle loop
  Value tileC;
  SmallVector<Value, 8U> partRes;
  SmallVector<Value, 8U> tileB;

  rewriter.setInsertionPointToStart(loops[1].getBody());
  if (minWrites) {
    for (unsigned i = 0; i < numCimTiles; ++i) {
      Value cimTileIter = createIndexConst(op, rewriter, i);
      Value cimTileIterK = rewriter.create<AddIOp>(
          op->getLoc(), rewriter.getIndexType(), iterK, cimTileIter);

      tileB.push_back(allocateTile(op, rewriter, matB, cimTileIterK, iterN,
                                   sizeTile, true));

      loop::IfOp ifOp;
      if (boundaryChecks) {
        ifOp = insertUnrollBoundaryCheck(op, rewriter, iterK, maxK, i);
      }

      rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), cimTileIds[i],
                                              tileB[i]);

      if (boundaryChecks) {
        exitIfBlock(op, rewriter, ifOp);
      }
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
      loop::IfOp ifOp;
      if (boundaryChecks) {
        ifOp = insertUnrollBoundaryCheck(op, rewriter, iterK, maxK, i);
      }

      if (isMatrix) {
        rewriter.create<cim::GemmOp>(op->getLoc(), cimTileIds[i], tileA[i],
                                     partRes[i]);
      } else {
        rewriter.create<cim::GevmOp>(op->getLoc(), cimTileIds[i], tileA[i],
                                     partRes[i]);
      }

      if (boundaryChecks) {
        exitIfBlock(op, rewriter, ifOp);
      }
    }

    for (unsigned i = 0; i < numCimTiles; ++i) {
      loop::IfOp ifOp;
      if (boundaryChecks) {
        ifOp = insertUnrollBoundaryCheck(op, rewriter, iterK, maxK, i);
      }

      rewriter.create<cim::BarrierOp>(op->getLoc(), cimTileIds[i]);
      elementwiseAddition(op, rewriter, partRes[i], tileC);

      if (boundaryChecks) {
        exitIfBlock(op, rewriter, ifOp);
      }
    }

    for (unsigned i = 0; i < numCimTiles; ++i) {
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

      loop::IfOp ifOp;
      if (boundaryChecks) {
        ifOp = insertUnrollBoundaryCheck(op, rewriter, iterK, maxK, i);
      }

      rewriter.create<cim::WriteToCrossbarOp>(op->getLoc(), cimTileIds[i],
                                              tileB[i]);

      if (boundaryChecks) {
        exitIfBlock(op, rewriter, ifOp);
      }
    }

    for (unsigned i = 0; i < numCimTiles; ++i) {
      loop::IfOp ifOp;
      if (boundaryChecks) {
        ifOp = insertUnrollBoundaryCheck(op, rewriter, iterK, maxK, i);
      }

      if (isMatrix) {
        rewriter.create<cim::GemmOp>(op->getLoc(), cimTileIds[i], tileA[i],
                                     partRes[i]);
      } else {
        rewriter.create<cim::GevmOp>(op->getLoc(), cimTileIds[i], tileA[i],
                                     partRes[i]);
      }

      if (boundaryChecks) {
        exitIfBlock(op, rewriter, ifOp);
      }
    }

    for (unsigned i = 0; i < numCimTiles; ++i) {
      loop::IfOp ifOp;
      if (boundaryChecks) {
        ifOp = insertUnrollBoundaryCheck(op, rewriter, iterK, maxK, i);
      }

      rewriter.create<cim::BarrierOp>(op->getLoc(), cimTileIds[i]);
      elementwiseAddition(op, rewriter, partRes[i], tileC);

      if (boundaryChecks) {
        exitIfBlock(op, rewriter, ifOp);
      }

      rewriter.create<DeallocOp>(op->getLoc(), tileB[i]);
    }
  }

  for (unsigned i = 0; i < numCimTiles; ++i) {
    rewriter.create<DeallocOp>(op->getLoc(), tileA[i]);
  }

  // Set insertion point back at main body outside of the loops
  rewriter.setInsertionPointAfter(loops.front());
}

void mlir::cim::createCIMTiledGEMM(Operation *op, PatternRewriter &rewriter,
                                   ConstantOp &tileId, const Value &matA,
                                   const Value &matB, const Value &matC,
                                   uint32_t tileSize, bool minWrites,
                                   unsigned numTilesOption) {
  auto dimsA = getMemRefSizes(op, rewriter, matA);
  auto dimsC = getMemRefSizes(op, rewriter, matC);

  bool isMatrix = dimsC.size() == 2;
  unsigned numCimTiles = numTilesOption == 0 ? 1 : numTilesOption;

  SmallVector<Value, 8U> cimTileIds;
  for (unsigned i = 0; i < numCimTiles; ++i) {
    cimTileIds.push_back(createIntConst(op, rewriter, i, 32));
  }

  Value dimM;
  Value dimN;
  Value dimK;
  if (isMatrix) {
    dimM = dimsC[0];
    dimN = dimsC[1];
    dimK = dimsA[1];
  } else {
    // As CIM natively supports only vector-matrix multiplication,
    // assume that the A and C operands are always row vectors (1,N)
    dimM = createIndexConst(op, rewriter, 1);
    dimN = dimsC[0];
    dimK = dimsA[0];
  }

  Value zero = createIndexConst(op, rewriter, 0);
  Value one = createIndexConst(op, rewriter, 1);
  Value sizeTile = createIndexConst(op, rewriter, tileSize);
  Value cimTilesCount = createIndexConst(op, rewriter, numCimTiles);

  Value tiledRows = calculateNumTiles(op, rewriter, sizeTile, dimM);
  Value tiledCols = calculateNumTiles(op, rewriter, sizeTile, dimN);
  Value numTiles = calculateNumTiles(op, rewriter, sizeTile, dimK);

  Value lowerBound = createIndexConst(op, rewriter, 0);

  SmallVector<Value, 8U> steps;
  if (numCimTiles > 1) {
    // unroll loop of the matmul inner dimension K
    if (minWrites) {
      // loop interchange
      steps = {cimTilesCount, one, one};
    } else {
      // original loop order
      steps = {one, one, cimTilesCount};
    }
  } else {
    steps = {one, one, one};
  }

  Value upperBoundDimK;
  if (numCimTiles > 1) {
    upperBoundDimK = rewriter.create<SubIOp>(
        op->getLoc(), rewriter.getIndexType(), numTiles, cimTilesCount);
  } else {
    upperBoundDimK = numTiles;
  }

  // C[M][N] = A[M][K] * B[K][N]
  SmallVector<Value, 8U> upperBounds;
  if (minWrites) {
    // Iterators: k, n, m
    upperBounds = {upperBoundDimK, tiledCols, tiledRows};
  } else {
    // Iterators: m, n, k
    upperBounds = {tiledRows, tiledCols, upperBoundDimK};
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

  // Set insertion point before the loops
  rewriter.setInsertionPoint(loops.front());

  Value kDimIter;
  if (numCimTiles > 1) {
    // Keep track of the iterator over the unrolled dimension
    MemRefType memType = MemRefType::Builder({1}, rewriter.getIntegerType(32));
    kDimIter = rewriter.create<AllocOp>(op->getLoc(), memType).getResult();

    Value initVal = rewriter.create<ConstantOp>(
        op->getLoc(), rewriter.getIntegerType(32),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), -1 * numCimTiles));
    rewriter.create<StoreOp>(op->getLoc(), initVal, kDimIter, ValueRange(zero));

    int dimKPosition = minWrites ? 0 : 2;
    rewriter.setInsertionPointToStart(loops[dimKPosition].getBody());
    Value intIter = rewriter.create<IndexCastOp>(
        op->getLoc(), loopIterators[dimKPosition],
        kDimIter.getType().cast<MemRefType>().getElementType());
    rewriter.create<StoreOp>(op->getLoc(), intIter, kDimIter, ValueRange(zero));
  }

  populateTiledGEMMLoops(op, rewriter, tileId, matA, matB, matC, sizeTile,
                         minWrites, numTilesOption, loops, loopIterators, false,
                         upperBounds);

  // If the loops are unrolled, calculate new lower bounds for computing
  // remaining tail
  if (numCimTiles > 1) {
    Value kDimIterVal =
        rewriter.create<LoadOp>(op->getLoc(), kDimIter, ValueRange(zero));
    Value kDimIterIndex = rewriter.create<IndexCastOp>(
        op->getLoc(), kDimIterVal, rewriter.getIndexType());

    SmallVector<Value, 8U> lowerBounds;
    Value kDimLowerBound = rewriter.create<AddIOp>(
        op->getLoc(), rewriter.getIndexType(), kDimIterIndex, cimTilesCount);
    if (minWrites) {
      // Iterators: k, n, m
      lowerBounds = {kDimLowerBound, zero, zero};
    } else {
      // Iterators: m, n, k
      lowerBounds = {zero, zero, kDimLowerBound};
    }

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
      auto loop = rewriter.create<loop::ForOp>(op->getLoc(), lowerBounds[i],
                                               upperBounds[i], steps[i]);
      loops.push_back(loop);
      loopIterators.push_back(loop.getInductionVar());

      // Set insertion point inside the loop
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Handle the tail with boundary checks inserted around every
    // unrolled computational operation
    populateTiledGEMMLoops(op, rewriter, tileId, matA, matB, matC, sizeTile,
                           minWrites, numTilesOption, loops, loopIterators,
                           true, upperBounds);

    rewriter.create<DeallocOp>(op->getLoc(), kDimIter);
  }
}