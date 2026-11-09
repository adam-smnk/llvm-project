//===- DecomposeVectorScaledContract.cpp - Decompose scaled_contract ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns decomposing vector.scaled_contract into
// standard vector operations. They serve as a generic fallback for targets
// without native scaled contraction support.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::vector;

/// Returns the type in which scaled values are accumulated. It is at least f32
/// such that narrow outputs are rounded only once.
static FloatType getComputeType(ScaledContractionOp op) {
  auto resType = cast<FloatType>(getElementTypeOrSelf(op.getResultType()));
  return resType.getWidth() > 32 ? resType : Float32Type::get(op.getContext());
}

/// Converts the elements of a scalar or vector `value` to `elemType`. Integers
/// are signed.
static Value convertElementType(OpBuilder &b, Location loc, Value value,
                                Type elemType) {
  Type srcElemType = getElementTypeOrSelf(value);
  if (srcElemType == elemType)
    return value;
  Type dstType = elemType;
  if (auto vectorType = dyn_cast<VectorType>(value.getType()))
    dstType = vectorType.clone(elemType);
  if (isa<IntegerType>(srcElemType))
    return arith::SIToFPOp::create(b, loc, dstType, value);
  if (srcElemType.getIntOrFloatBitWidth() < elemType.getIntOrFloatBitWidth())
    return arith::ExtFOp::create(b, loc, dstType, value,
                                 arith::FastMathFlagsAttr{});
  return arith::TruncFOp::create(b, loc, dstType, value);
}

/// Reshapes `value` into `shape`. An empty shape yields a scalar.
static Value reshape(OpBuilder &b, Location loc, Value value,
                     ArrayRef<int64_t> shape) {
  auto vectorType = dyn_cast<VectorType>(value.getType());
  if (!vectorType || vectorType.getShape() == shape)
    return value;
  if (shape.empty())
    return vector::ExtractOp::create(
        b, loc, value, SmallVector<int64_t>(vectorType.getRank(), 0));
  return vector::ShapeCastOp::create(
      b, loc, VectorType::get(shape, vectorType.getElementType()), value);
}

/// Broadcasts `value` indexed by `map` to `targetShape` indexed by `targetMap`.
/// All results of `map` must be results of `targetMap`.
static Value broadcastToLayout(OpBuilder &b, Location loc, Value value,
                               AffineMap map, AffineMap targetMap,
                               ArrayRef<int64_t> targetShape) {
  if (targetShape.empty())
    return value;
  auto targetType = VectorType::get(targetShape, getElementTypeOrSelf(value));
  if (map.getNumResults() == 0)
    return vector::BroadcastOp::create(b, loc, targetType, value);

  SmallVector<int64_t> targetPositions;
  for (AffineExpr expr : map.getResults())
    targetPositions.push_back(*targetMap.getResultPosition(expr));
  SmallVector<int64_t> perm =
      llvm::to_vector(llvm::seq<int64_t>(map.getNumResults()));
  llvm::sort(perm, [&](int64_t lhs, int64_t rhs) {
    return targetPositions[lhs] < targetPositions[rhs];
  });
  if (!llvm::is_sorted(targetPositions))
    value = vector::TransposeOp::create(b, loc, value, perm);

  SmallVector<int64_t> unitShape(targetShape.size(), 1);
  for (int64_t pos : targetPositions)
    unitShape[pos] = targetShape[pos];
  value = reshape(b, loc, value, unitShape);
  if (value.getType() == targetType)
    return value;
  return vector::BroadcastOp::create(b, loc, targetType, value);
}

/// Returns the shape indexed by the results of a projected permutation `map`.
static SmallVector<int64_t> getResultShape(AffineMap map,
                                           ArrayRef<int64_t> dimSizes) {
  return llvm::map_to_vector(map.getResults(), [&](AffineExpr expr) {
    return dimSizes[cast<AffineDimExpr>(expr).getPosition()];
  });
}

/// Returns a vector type of `shape`, or `elemType` for an empty shape.
static Type getShapedType(ArrayRef<int64_t> shape, Type elemType) {
  if (shape.empty())
    return elemType;
  return VectorType::get(shape, elemType);
}

namespace {
/// Iteration space of a scaled contraction where each block scaled dim is
/// split into block levels, e.g., `k` with scale map `k floordiv 32` becomes
/// `(ko, kb)`. All operand maps become projected permutations.
struct BlockedIterationSpace {
  SmallVector<int64_t> dimSizes;
  SmallVector<IteratorType> iteratorTypes;
  /// Indexing maps of the op's operands in the new iteration space.
  SmallVector<AffineMap> maps;
};
} // namespace

static FailureOr<BlockedIterationSpace>
splitBlockScaledDims(ScaledContractionOp op) {
  MLIRContext *ctx = op.getContext();
  SmallVector<IteratorType> iteratorTypes = op.getIteratorTypesArray();
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  unsigned numDims = iteratorTypes.size();

  // A block spanning the whole dim has a single scale and needs no split.
  auto isSingleBlock = [&](unsigned dim, int64_t blockSize) {
    return loopRanges[dim] <= blockSize;
  };
  SmallVector<SmallVector<int64_t>> blockSizes(numDims);
  for (OpOperand *scale : op.getScaleOperands()) {
    for (AffineExpr expr : op.getMatchingIndexingMap(scale).getResults()) {
      auto [dim, blockSize] =
          *ScaledContractionOp::getScaleDimAndBlockSize(expr);
      if (blockSize > 1 && !isSingleBlock(dim, blockSize))
        blockSizes[dim].push_back(blockSize);
    }
  }

  // Sizes of block levels from outermost to innermost, e.g., [K / 64, 2, 32]
  // for block sizes 32 and 64. Different block sizes of a dim must nest.
  SmallVector<SmallVector<int64_t>> levelSizes(numDims);
  for (unsigned dim = 0; dim < numDims; ++dim) {
    SmallVector<int64_t> &blocks = blockSizes[dim];
    if (blocks.empty()) {
      levelSizes[dim].push_back(loopRanges[dim]);
      continue;
    }
    llvm::sort(blocks);
    blocks.erase(llvm::unique(blocks), blocks.end());
    for (unsigned i = 1; i < blocks.size(); ++i)
      if (blocks[i] % blocks[i - 1] != 0)
        return failure();
    levelSizes[dim].push_back(loopRanges[dim] / blocks.back());
    for (unsigned i = blocks.size() - 1; i > 0; --i)
      levelSizes[dim].push_back(blocks[i] / blocks[i - 1]);
    levelSizes[dim].push_back(blocks.front());
  }

  BlockedIterationSpace space;
  SmallVector<SmallVector<AffineExpr>> newDims(numDims);
  for (unsigned dim = 0; dim < numDims; ++dim) {
    for (int64_t size : levelSizes[dim]) {
      newDims[dim].push_back(getAffineDimExpr(space.dimSizes.size(), ctx));
      space.dimSizes.push_back(size);
      space.iteratorTypes.push_back(iteratorTypes[dim]);
    }
  }

  SmallVector<OpOperand *> scales = op.getScaleOperands();
  for (OpOperand &operand : op->getOpOperands()) {
    bool isScale = llvm::is_contained(scales, &operand);
    SmallVector<AffineExpr> results;
    for (AffineExpr expr : op.getMatchingIndexingMap(&operand).getResults()) {
      auto [dim, blockSize] =
          isScale ? *ScaledContractionOp::getScaleDimAndBlockSize(expr)
                  : std::make_pair(cast<AffineDimExpr>(expr).getPosition(),
                                   int64_t{1});
      if (blockSize == 1) {
        llvm::append_range(results, newDims[dim]);
        continue;
      }
      if (isSingleBlock(dim, blockSize))
        continue;
      // A block scale indexes all levels coarser than the block.
      int64_t numScales = loopRanges[dim] / blockSize;
      for (auto [levelExpr, levelSize] :
           llvm::zip_equal(newDims[dim], levelSizes[dim])) {
        if (numScales == 1)
          break;
        results.push_back(levelExpr);
        numScales /= levelSize;
      }
    }
    space.maps.push_back(
        AffineMap::get(space.dimSizes.size(), 0, results, ctx));
  }
  return space;
}

namespace {
/// Decomposes a scaled contraction following the OCP Microscaling dot product
/// definition:
///
/// 1. Block scaled dims are split into block levels such that each scale is
///    constant within the innermost levels.
/// 2. The unscaled inputs are contracted over the reduction dims not used by
///    the scales:
///      T = SUM(A * B)
/// 3. The partial results are scaled and reduced over the remaining reduction
///    dims into the accumulator:
///      D = SUM((T * scale_A) * scale_B) + C
///
/// Preserves a low-precision contraction of the inputs which can be lowered
/// separately to efficient hardware instructions.
struct DecomposeScaledContractByBlocks
    : public OpRewritePattern<ScaledContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ScaledContractionOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<BlockedIterationSpace> space = splitBlockScaledDims(op);
    if (failed(space))
      return rewriter.notifyMatchFailure(op, "block sizes do not nest");
    ArrayRef<int64_t> dimSizes = space->dimSizes;
    AffineMap lhsMap = space->maps[op.getLhsMutable().getOperandNumber()];
    AffineMap rhsMap = space->maps[op.getRhsMutable().getOperandNumber()];
    AffineMap accMap = space->maps[op.getAccMutable().getOperandNumber()];
    SmallVector<AffineMap> scaleMaps = {
        space->maps[op.getLhsScaleMutable().getOperandNumber()],
        space->maps[op.getRhsScaleMutable().getOperandNumber()]};

    // Reduction dims used by the scales are kept in the partial results.
    MLIRContext *ctx = op.getContext();
    SmallVector<AffineExpr> partialExprs(accMap.getResults());
    SmallVector<Attribute> contractIteratorTypes;
    bool hasContractedDim = false;
    for (auto [dim, iteratorType] : llvm::enumerate(space->iteratorTypes)) {
      bool isUsedByScale =
          llvm::any_of(scaleMaps, [&, dim = dim](AffineMap map) {
            return map.isFunctionOfDim(dim);
          });
      bool isContracted =
          iteratorType == IteratorType::reduction && !isUsedByScale;
      if (iteratorType == IteratorType::reduction && isUsedByScale)
        partialExprs.push_back(getAffineDimExpr(dim, ctx));
      hasContractedDim |= isContracted;
      contractIteratorTypes.push_back(
          IteratorTypeAttr::get(ctx, isContracted ? IteratorType::reduction
                                                  : IteratorType::parallel));
    }
    if (!hasContractedDim)
      return rewriter.notifyMatchFailure(op, "no dims to contract");
    AffineMap partialMap =
        AffineMap::get(dimSizes.size(), 0, partialExprs, ctx);
    SmallVector<int64_t> partialShape = getResultShape(partialMap, dimSizes);

    Location loc = op.getLoc();
    Type lhsElemType = op.getLhsType().getElementType();
    Type rhsElemType = op.getRhsType().getElementType();
    FloatType computeType = getComputeType(op);

    // Narrow integers are accumulated exactly in i32.
    auto intType = dyn_cast<IntegerType>(lhsElemType);
    bool isNarrowInt = intType && intType.getWidth() <= 8;
    Type partialElemType =
        isNarrowInt ? cast<Type>(rewriter.getI32Type()) : computeType;

    Value lhs =
        reshape(rewriter, loc, op.getLhs(), getResultShape(lhsMap, dimSizes));
    Value rhs =
        reshape(rewriter, loc, op.getRhs(), getResultShape(rhsMap, dimSizes));
    // vector.contract requires inputs of the same type and only promotes
    // integers to wider integers.
    if (lhsElemType != rhsElemType || (intType && !isNarrowInt)) {
      lhs = convertElementType(rewriter, loc, lhs, partialElemType);
      rhs = convertElementType(rewriter, loc, rhs, partialElemType);
    }
    Type partialType = getShapedType(partialShape, partialElemType);
    Value zero = arith::ConstantOp::create(rewriter, loc, partialType,
                                           rewriter.getZeroAttr(partialType));
    Value partial = vector::ContractionOp::create(
        rewriter, loc, lhs, rhs, zero,
        rewriter.getAffineMapArrayAttr({lhsMap, rhsMap, partialMap}),
        rewriter.getArrayAttr(contractIteratorTypes));

    Value scaled = convertElementType(rewriter, loc, partial, computeType);
    for (auto [scale, scaleMap] :
         llvm::zip_equal(op.getScaleOperands(), scaleMaps)) {
      Value scaleValue = reshape(rewriter, loc, scale->get(),
                                 getResultShape(scaleMap, dimSizes));
      scaleValue = convertElementType(rewriter, loc, scaleValue, computeType);
      scaleValue = broadcastToLayout(rewriter, loc, scaleValue, scaleMap,
                                     partialMap, partialShape);
      scaled = arith::MulFOp::create(rewriter, loc, scaled, scaleValue);
    }

    // Accumulate in the compute type such that the result is rounded once.
    SmallVector<int64_t> accShape = getResultShape(accMap, dimSizes);
    Value acc = reshape(rewriter, loc, op.getAcc(), accShape);
    Type resElemType = getElementTypeOrSelf(op.getResultType());
    SmallVector<bool> reductionMask(partialShape.size(), false);
    std::fill(reductionMask.begin() + accShape.size(), reductionMask.end(),
              true);
    Value result;
    if (accShape.size() == partialShape.size()) {
      result = arith::AddFOp::create(
          rewriter, loc, convertElementType(rewriter, loc, acc, computeType),
          scaled);
    } else if (resElemType == computeType) {
      result = vector::MultiDimReductionOp::create(
          rewriter, loc, scaled, acc, reductionMask, CombiningKind::ADD);
    } else {
      Type sumType = getShapedType(accShape, computeType);
      Value sumInit = arith::ConstantOp::create(rewriter, loc, sumType,
                                                rewriter.getZeroAttr(sumType));
      Value sum = vector::MultiDimReductionOp::create(
          rewriter, loc, scaled, sumInit, reductionMask, CombiningKind::ADD);
      result = arith::AddFOp::create(
          rewriter, loc, convertElementType(rewriter, loc, acc, computeType),
          sum);
    }
    result = convertElementType(rewriter, loc, result, resElemType);
    if (result.getType() != op.getResultType())
      result = vector::ShapeCastOp::create(rewriter, loc, op.getResultType(),
                                           result);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Decomposes a scaled contraction into scaling of the inputs followed by a
/// contraction of the scaled inputs in the compute type:
///
///   D = SUM((A * scale_A) * (B * scale_B)) + C
///
/// Applies to all scaling schemes, e.g., when every element along a reduction
/// dim has its own scale, and thus, it serves as a fallback.
struct DecomposeScaledContractByPrescaling
    : public OpRewritePattern<ScaledContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ScaledContractionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    FloatType computeType = getComputeType(op);
    SmallVector<Value> scaledInputs;
    for (auto [data, scale] :
         llvm::zip_equal(op.getDataOperands(), op.getScaleOperands())) {
      Value dataValue =
          convertElementType(rewriter, loc, data->get(), computeType);
      Value scaleValue =
          convertElementType(rewriter, loc, scale->get(), computeType);
      scaleValue = expandScale(rewriter, loc, scaleValue,
                               op.getMatchingIndexingMap(scale),
                               op.getMatchingIndexingMap(data),
                               cast<VectorType>(dataValue.getType()));
      scaledInputs.push_back(
          arith::MulFOp::create(rewriter, loc, dataValue, scaleValue));
    }

    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    Value acc = convertElementType(rewriter, loc, op.getAcc(), computeType);
    Value result = vector::ContractionOp::create(
        rewriter, loc, scaledInputs[0], scaledInputs[1], acc,
        rewriter.getAffineMapArrayAttr(
            {maps[op.getLhsMutable().getOperandNumber()],
             maps[op.getRhsMutable().getOperandNumber()],
             maps[op.getAccMutable().getOperandNumber()]}),
        op.getIteratorTypes());
    result = convertElementType(rewriter, loc, result,
                                getElementTypeOrSelf(op.getResultType()));
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  /// Expands `scale` indexed by `scaleMap` to the layout of its data operand
  /// such that each data element is paired with its scale.
  static Value expandScale(OpBuilder &b, Location loc, Value scale,
                           AffineMap scaleMap, AffineMap dataMap,
                           VectorType dataType) {
    if (scaleMap.getNumResults() == 0)
      return vector::BroadcastOp::create(b, loc, dataType, scale);

    // Order the scale dims as in the data.
    SmallVector<std::tuple<int64_t, int64_t, int64_t>> dims;
    for (auto [pos, expr] : llvm::enumerate(scaleMap.getResults())) {
      auto [dim, blockSize] =
          *ScaledContractionOp::getScaleDimAndBlockSize(expr);
      int64_t dataPos =
          *dataMap.getResultPosition(getAffineDimExpr(dim, b.getContext()));
      dims.emplace_back(dataPos, pos, blockSize);
    }
    llvm::sort(dims);
    SmallVector<int64_t> perm =
        llvm::map_to_vector(dims, [](auto dim) { return std::get<1>(dim); });
    if (!llvm::is_sorted(perm))
      scale = vector::TransposeOp::create(b, loc, scale, perm);
    ArrayRef<int64_t> scaleShape = cast<VectorType>(scale.getType()).getShape();

    // Insert a unit dim for each scaled block and each dim without scales,
    // then broadcast the scales across these dims.
    SmallVector<int64_t> unitShape;
    SmallVector<int64_t> broadcastShape;
    auto *dimIt = dims.begin();
    for (auto [dataPos, dataSize] : llvm::enumerate(dataType.getShape())) {
      if (dimIt == dims.end() ||
          std::get<0>(*dimIt) != static_cast<int64_t>(dataPos)) {
        unitShape.push_back(1);
        broadcastShape.push_back(dataSize);
        continue;
      }
      int64_t scaleSize = scaleShape[std::distance(dims.begin(), dimIt)];
      int64_t blockSize = std::get<2>(*dimIt);
      ++dimIt;
      if (blockSize == 1 || scaleSize == 1) {
        unitShape.push_back(scaleSize);
        broadcastShape.push_back(dataSize);
        continue;
      }
      unitShape.append({scaleSize, 1});
      broadcastShape.append({scaleSize, blockSize});
    }
    scale = reshape(b, loc, scale, unitShape);
    auto broadcastType =
        VectorType::get(broadcastShape, dataType.getElementType());
    if (scale.getType() != broadcastType)
      scale = vector::BroadcastOp::create(b, loc, broadcastType, scale);
    return reshape(b, loc, scale, dataType.getShape());
  }
};
} // namespace

void mlir::vector::populateVectorScaledContractDecompositionPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<DecomposeScaledContractByBlocks>(patterns.getContext(),
                                                benefit.getBenefit() + 1);
  patterns.add<DecomposeScaledContractByPrescaling>(patterns.getContext(),
                                                    benefit);
}
