#include "mlir/Dialect/CIM/Utils/RuntimeUtils.h"
#include "mlir/Dialect/CIM/IR/CIMDialect.h"
#include "mlir/Dialect/CIM/Utils/StaticUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::cim;

Value mlir::cim::transposeMemRef(Operation *op, PatternRewriter &rewriter,
                                 const Value &memRef,
                                 const AffineMap &memRefMap,
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

SmallVector<Value, 8U>
mlir::cim::getMemRefSizes(Operation *op, PatternRewriter &rewriter,
                          const Value &memRef,
                          const ArrayRef<unsigned> &dimsPositions) {
  SmallVector<Value, 8U> dimOperands;

  if (!dimsPositions.empty()) {
    for (const auto &pos : dimsPositions) {
      dimOperands.push_back(rewriter.create<DimOp>(op->getLoc(), memRef, pos));
    }
  } else {
    int numDims = memRef.getType().cast<MemRefType>().getShape().size();

    for (int i = 0; i < numDims; ++i) {
      dimOperands.push_back(rewriter.create<DimOp>(op->getLoc(), memRef, i));
    }
  }

  return dimOperands;
}

Value mlir::cim::calculateDimsSize(Operation *op, PatternRewriter &rewriter,
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

SmallVector<Value, 8U>
mlir::cim::calculateDimsStrides(Operation *op, PatternRewriter &rewriter,
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
Value mlir::cim::calculateLinearIndex(Operation *op, PatternRewriter &rewriter,
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

Value mlir::cim::allocateDuplicate(Operation *op, PatternRewriter &rewriter,
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

void mlir::cim::elementwiseAddition(Operation *op, PatternRewriter &rewriter,
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
  rewriter.setInsertionPointAfter(loops.front());
}

// Groups (collapses) dimensions of the input memref based on a reassociation
// map. Allows for permutation.
// Performs alloc and copy of the original data.
Value mlir::cim::groupDimensions(Operation *op, PatternRewriter &rewriter,
                                 const Value &memRef,
                                 const AffineMap &memRefMap,
                                 const ArrayAttr &reassociation) {
  auto *ctx = op->getContext();

  SmallVector<AffineMap, 8U> targetMaps = getResultMaps(reassociation);

  AffineMap targetMap = combineMaps(targetMaps);
  SmallVector<AffineExpr, 8U> outputPermutation =
      getPermutation(memRefMap, targetMap, ctx);

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

// Performs reshape with copy.
// Reassociation defines continuous dimension groupings for the lower rank
// memref.
// Expects both memrefs to point to contiguous memory.
// Permutation list should reorder all the dimensions present in the higher rank
// memref or be empty, otherwise the behavior is undefined.
void mlir::cim::reshapeCopy(Operation *op, PatternRewriter &rewriter,
                            const Value &inputMemRef, const Value &outputMemRef,
                            const ArrayAttr &reassociation,
                            ArrayRef<unsigned> permutation,
                            bool performElementwiseSum) {
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
  rewriter.setInsertionPointAfter(loops.front());
}

Value mlir::cim::createIndexConst(Operation *op, PatternRewriter &rewriter,
                                  int64_t value) {
  return rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), value));
}

Value mlir::cim::minSigned(Operation *op, PatternRewriter &rewriter,
                           const Value &lhs, const Value &rhs) {
  Value cond =
      rewriter.create<CmpIOp>(op->getLoc(), CmpIPredicate::slt, lhs, rhs);
  return rewriter.create<SelectOp>(op->getLoc(), cond, lhs, rhs);
}

Value mlir::cim::maxSigned(Operation *op, PatternRewriter &rewriter,
                           const Value &lhs, const Value &rhs) {
  Value cond =
      rewriter.create<CmpIOp>(op->getLoc(), CmpIPredicate::sgt, lhs, rhs);
  return rewriter.create<SelectOp>(op->getLoc(), cond, lhs, rhs);
}

Value mlir::cim::minUnsigned(Operation *op, PatternRewriter &rewriter,
                             const Value &lhs, const Value &rhs) {
  Value cond =
      rewriter.create<CmpIOp>(op->getLoc(), CmpIPredicate::ult, lhs, rhs);
  return rewriter.create<SelectOp>(op->getLoc(), cond, lhs, rhs);
}

Value mlir::cim::maxUnsigned(Operation *op, PatternRewriter &rewriter,
                             const Value &lhs, const Value &rhs) {
  Value cond =
      rewriter.create<CmpIOp>(op->getLoc(), CmpIPredicate::ugt, lhs, rhs);
  return rewriter.create<SelectOp>(op->getLoc(), cond, lhs, rhs);
}

// Calculates number of tiles rounded up
Value mlir::cim::calculateNumTiles(Operation *op, PatternRewriter &rewriter,
                                   const Value &tileSize,
                                   const Value &dimMaxSize) {
  const auto resType = rewriter.getIndexType();

  Value numFullTiles = rewriter.create<UnsignedDivIOp>(op->getLoc(), resType,
                                                       dimMaxSize, tileSize);
  Value numPartialTiles = rewriter.create<UnsignedRemIOp>(op->getLoc(), resType,
                                                          dimMaxSize, tileSize);

  Value one = createIndexConst(op, rewriter, 1);
  Value numFullTilesPlusOne =
      rewriter.create<AddIOp>(op->getLoc(), resType, numFullTiles, one);

  Value zero = createIndexConst(op, rewriter, 0);
  Value cond = rewriter.create<CmpIOp>(op->getLoc(), CmpIPredicate::eq,
                                       numPartialTiles, zero);
  return rewriter.create<SelectOp>(op->getLoc(), cond, numFullTiles,
                                   numFullTilesPlusOne);
}

static Value calculateTileStartIndex(Operation *op, PatternRewriter &rewriter,
                                     const Value &tilePos,
                                     const Value &tileSize,
                                     const Value &dimMaxSize) {
  const auto resType = rewriter.getIndexType();
  Value tileStartPos =
      rewriter.create<MulIOp>(op->getLoc(), resType, tilePos, tileSize);

  return minUnsigned(op, rewriter, tileStartPos, dimMaxSize);
}

static Value calculateTileEndIndex(Operation *op, PatternRewriter &rewriter,
                                   const Value &tilePos, const Value &tileSize,
                                   const Value &dimMaxSize) {
  Value one = createIndexConst(op, rewriter, 1);

  const auto resType = rewriter.getIndexType();
  Value nextTilePos =
      rewriter.create<AddIOp>(op->getLoc(), resType, tilePos, one);
  Value tileEndPos =
      rewriter.create<MulIOp>(op->getLoc(), resType, nextTilePos, tileSize);

  return minUnsigned(op, rewriter, tileEndPos, dimMaxSize);
}

// Supports both vectors and matrices
Value mlir::cim::allocateTile(Operation *op, PatternRewriter &rewriter,
                              const Value &mat, const Value &row,
                              const Value &col, const Value &tileSize,
                              bool copyData) {
  auto indexType = rewriter.getIndexType();
  SmallVector<Value, 8U> allocOperands;

  int numDims = mat.getType().cast<MemRefType>().getShape().size();
  bool isMatrix = numDims == 2;

  // Tile size is limited by the matrix boundaries
  Value startRow;
  if (isMatrix) {
    Value matRows = rewriter.create<DimOp>(op->getLoc(), mat, 0);
    startRow = calculateTileStartIndex(op, rewriter, row, tileSize, matRows);
    Value endRow = calculateTileEndIndex(op, rewriter, row, tileSize, matRows);
    Value numRows =
        rewriter.create<SubIOp>(op->getLoc(), indexType, endRow, startRow);
    allocOperands.push_back(numRows);
  }

  Value matCols = rewriter.create<DimOp>(op->getLoc(), mat, numDims - 1);
  Value startCol =
      calculateTileStartIndex(op, rewriter, col, tileSize, matCols);
  Value endCol = calculateTileEndIndex(op, rewriter, col, tileSize, matCols);
  Value numCols =
      rewriter.create<SubIOp>(op->getLoc(), indexType, endCol, startCol);
  allocOperands.push_back(numCols);

  SmallVector<int64_t, 8U> tileSizes;
  for (int i = 0; i < numDims; ++i) {
    tileSizes.push_back(-1);
  }

  MemRefType tileType = MemRefType::Builder(
      tileSizes, mat.getType().cast<MemRefType>().getElementType());
  Value tile = rewriter.create<AllocOp>(op->getLoc(), tileType, allocOperands)
                   .getResult();

  if (copyData) {
    // Copy data from the matrix to the tile
    Value lowerBound = createIndexConst(op, rewriter, 0);
    Value step = createIndexConst(op, rewriter, 1);
    SmallVector<Value, 8U> upperBounds = allocOperands;

    SmallVector<loop::ForOp, 8U> loops;
    SmallVector<Value, 8U> loopIterators;
    for (const Value &ub : upperBounds) {
      auto loop =
          rewriter.create<loop::ForOp>(op->getLoc(), lowerBound, ub, step);
      loops.push_back(loop);
      loopIterators.push_back(loop.getInductionVar());

      // Set insertion point inside the loop
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    SmallVector<Value, 8U> matPos;

    if (isMatrix) {
      Value matRowPos = rewriter.create<AddIOp>(op->getLoc(), indexType,
                                                startRow, loopIterators[0]);
      matPos.push_back(matRowPos);
    }

    Value matColPos = rewriter.create<AddIOp>(op->getLoc(), indexType, startCol,
                                              loopIterators[numDims - 1]);
    matPos.push_back(matColPos);

    Value matElement =
        rewriter.create<LoadOp>(op->getLoc(), mat, ValueRange(matPos));
    rewriter.create<StoreOp>(op->getLoc(), matElement, tile,
                             ValueRange(loopIterators));

    // Set insertion point back at main body outside of the loops
    rewriter.setInsertionPointAfter(loops.front());
  } else {
    // Zero-initialize the tile
    Value zero = rewriter.create<ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(
            tile.getType().cast<MemRefType>().getElementType(), 0));
    rewriter.create<FillOp>(op->getLoc(), tile, zero);
  }

  return tile;
}

// Supports both vectors and matrices
void mlir::cim::storeTile(Operation *op, PatternRewriter &rewriter,
                          const Value &tile, const Value &mat, const Value &row,
                          const Value &col, const Value &tileSize) {
  auto indexType = rewriter.getIndexType();
  int numDims = mat.getType().cast<MemRefType>().getShape().size();
  bool isMatrix = numDims == 2;

  Value lowerBound = createIndexConst(op, rewriter, 0);
  Value step = createIndexConst(op, rewriter, 1);
  SmallVector<Value, 8U> upperBounds;

  Value startRow;
  if (isMatrix) {
    Value tileRows = rewriter.create<DimOp>(op->getLoc(), tile, 0);
    startRow = rewriter.create<MulIOp>(op->getLoc(), indexType, row, tileSize);
    upperBounds.push_back(tileRows);
  }

  Value tileCols = rewriter.create<DimOp>(op->getLoc(), tile, numDims - 1);
  Value startCol =
      rewriter.create<MulIOp>(op->getLoc(), indexType, col, tileSize);
  upperBounds.push_back(tileCols);

  SmallVector<loop::ForOp, 8U> loops;
  SmallVector<Value, 8U> loopIterators;
  for (const Value &ub : upperBounds) {
    auto loop =
        rewriter.create<loop::ForOp>(op->getLoc(), lowerBound, ub, step);
    loops.push_back(loop);
    loopIterators.push_back(loop.getInductionVar());

    // Set insertion point inside the loop
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  SmallVector<Value, 8U> matPos;

  if (isMatrix) {
    Value matRowPos = rewriter.create<AddIOp>(op->getLoc(), indexType, startRow,
                                              loopIterators[0]);
    matPos.push_back(matRowPos);
  }

  Value matColPos = rewriter.create<AddIOp>(op->getLoc(), indexType, startCol,
                                            loopIterators[numDims - 1]);
  matPos.push_back(matColPos);

  Value tileElement =
      rewriter.create<LoadOp>(op->getLoc(), tile, ValueRange(loopIterators));
  rewriter.create<StoreOp>(op->getLoc(), tileElement, mat, ValueRange(matPos));

  // Set insertion point back at main body outside of the loops
  rewriter.setInsertionPointAfter(loops.front());
}
