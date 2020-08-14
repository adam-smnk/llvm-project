//===- GEMMTiling.cpp - Implementation of CIM GEMM Tiling -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIM dialect GEMM Tiling pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CIM/IR/CIMDialect.h"
#include "mlir/Dialect/CIM/Utils/RuntimeUtils.h"
#include "mlir/Dialect/CIM/Utils/StaticUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LoopOps/EDSC/Builders.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/FoldUtils.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::loop;
using namespace mlir::cim;

#define DEBUG_TYPE "cim-gemm-tiling"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");
static llvm::cl::list<unsigned> clTileSizes(
    "cim-tile-sizes",
    llvm::cl::desc("CIM device tile sizes used to tile cim operations"),
    llvm::cl::OneOrMore, llvm::cl::MiscFlags::CommaSeparated,
    llvm::cl::cat(clOptionsCategory));

template <typename LoopTy>
Optional<TiledLinalgOp> static tileLinalgOpImpl(OpBuilder &b, LinalgOp op,
                                                ArrayRef<Value> tileSizes,
                                                ArrayRef<unsigned> permutation,
                                                OperationFolder *folder) {
  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");
  // 1. Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  assert(op.getNumParallelLoops() + op.getNumReductionLoops() +
                 op.getNumWindowLoops() ==
             tileSizes.size() &&
         "expected matching number of tile sizes and loops");

  // If permutation is empty, use the identity. Build the permutation map
  // otherwise.
  auto invPermutationMap = AffineMap::getMultiDimIdentityMap(
      tileSizes.size(), ScopedContext::getContext());
  if (!permutation.empty())
    invPermutationMap = inversePermutation(
        AffineMap::getPermutationMap(permutation, ScopedContext::getContext()));

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  ScopedContext scope(b, op.getLoc());
  // 2. Build the tiled loop ranges.
  auto viewSizes = getViewSizes(b, op);
  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (asserted in the inverse calculation).
  auto mapsRange = op.indexing_maps().getAsRange<AffineMapAttr>();
  auto maps =
      functional::map([](AffineMapAttr a) { return a.getValue(); }, mapsRange);
  auto viewSizesToLoopsMap = inversePermutation(concatAffineMaps(maps));
  assert(viewSizesToLoopsMap && "expected invertible map");

  SmallVector<SubViewOp::Range, 4> loopRanges;
  LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  std::tie(loopRanges, loopIndexToRangeIndex) =
      makeTiledLoopRanges(b, scope.getLocation(), viewSizesToLoopsMap,
                          viewSizes, tileSizes, folder);
  if (!permutation.empty())
    applyPermutationToVector(loopRanges, permutation);

  // 3. Create the tiled loops.
  LinalgOp res = op;
  auto ivs = ValueHandle::makeIndexHandles(loopRanges.size());
  auto pivs = makeHandlePointers(MutableArrayRef<ValueHandle>(ivs));
  // Convert SubViewOp::Range to linalg_range.
  SmallVector<Value, 4> linalgRanges;
  for (auto &range : loopRanges) {
    linalgRanges.push_back(
        linalg_range(range.offset, range.size, range.stride));
  }
  GenericLoopNestRangeBuilder<LoopTy>(pivs, linalgRanges)([&] {
    auto b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    SmallVector<Value, 4> ivValues(ivs.begin(), ivs.end());

    // If we have to apply a permutation to the tiled loop nest, we have to
    // reorder the induction variables This permutation is the right one
    // assuming that loopRanges have previously been permuted by
    // (i,j,k)->(k,i,j) So this permutation should be the inversePermutation of
    // that one: (d0,d1,d2)->(d2,d0,d1)
    if (!permutation.empty())
      ivValues = applyMapToValues(b, loc, invPermutationMap, ivValues, folder);

    auto views =
        makeTiledViews(b, loc, op, ivValues, tileSizes, viewSizes, folder);
    auto operands = getAssumedNonViewOperands(op);
    views.append(operands.begin(), operands.end());
    res = op.clone(b, loc, views);
  });

  // 4. Transforms index arguments of `linalg.generic` w.r.t. to the tiling.
  transformIndexedGenericOpIndices(b, res, pivs, loopIndexToRangeIndex);

  // 5. Gather the newly created loops and return them with the new op.
  SmallVector<Operation *, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs)
    loops.push_back(loop::getForInductionVarOwner(iv));

  return TiledLinalgOp{res, loops};
}

template <typename LoopTy>
Optional<TiledLinalgOp>
tileLinalgOpImpl(OpBuilder &b, LinalgOp op, ArrayRef<int64_t> tileSizes,
                 ArrayRef<unsigned> permutation, OperationFolder *folder) {
  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");
  if (tileSizes.empty())
    return llvm::None;

  // The following uses the convention that "tiling by zero" skips tiling a
  // particular dimension. This convention is significantly simpler to handle
  // instead of adjusting affine maps to account for missing dimensions.
  auto nLoops = op.getNumParallelLoops() + op.getNumReductionLoops() +
                op.getNumWindowLoops();
  tileSizes = tileSizes.take_front(nLoops);
  // If only 0 tilings are left, then return.
  if (llvm::all_of(tileSizes, [](int64_t v) { return v == 0; }))
    return llvm::None;

  // Create a builder for tile size constants.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  ScopedContext scope(b, op.getLoc());

  // Materialize concrete tile size values to pass the generic tiling function.
  SmallVector<Value, 8> tileSizeValues;
  tileSizeValues.reserve(tileSizes.size());
  for (auto ts : tileSizes)
    tileSizeValues.push_back(folded_std_constant_index(folder, ts));
  // Pad tile sizes with zero values to enforce our convention.
  if (tileSizeValues.size() < nLoops) {
    for (unsigned i = tileSizeValues.size(); i < nLoops; ++i)
      tileSizeValues.push_back(folded_std_constant_index(folder, 0));
  }

  return tileLinalgOpImpl<LoopTy>(b, op, tileSizeValues, permutation, folder);
}

Optional<TiledLinalgOp>
mlir::linalg::tileLinalgOp(OpBuilder &b, LinalgOp op, ArrayRef<Value> tileSizes,
                           ArrayRef<unsigned> permutation,
                           OperationFolder *folder) {
  return tileLinalgOpImpl<loop::ForOp>(b, op, tileSizes, permutation, folder);
}

Optional<TiledLinalgOp> mlir::linalg::tileLinalgOp(
    OpBuilder &b, LinalgOp op, ArrayRef<int64_t> tileSizes,
    ArrayRef<unsigned> permutation, OperationFolder *folder) {
  return tileLinalgOpImpl<loop::ForOp>(b, op, tileSizes, permutation, folder);
}

template <typename LoopTy>
static void tileLinalgOps(FuncOp f, ArrayRef<int64_t> tileSizes) {
  OpBuilder b(f);
  f.walk([tileSizes, &b](CimOp op) {
    auto opLoopsPair = tileLinalgOpImpl<LoopTy>(b, op, tileSizes);
    // If tiling occurred successfully, erase old op.
    if (opLoopsPair)
      op.erase();
  });
}

namespace {

template <typename LoopTy>
struct CIMGEMMTilingPass : public FunctionPass<CIMGEMMTilingPass<LoopTy>> {
  CIMGEMMTilingPass() = default;
  CIMGEMMTilingPass(ArrayRef<int64_t> sizes) {
    this->tileSizes.assign(sizes.begin(), sizes.end());
  }

  void runOnFunction() override {
    tileLinalgOps<LoopTy>(this->getFunction(), tileSizes);
  }

  SmallVector<int64_t, 8> tileSizes;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createCIMGEMMTilingPass(ArrayRef<int64_t> tileSizes) {
  return std::make_unique<CIMGEMMTilingPass<loop::ForOp>>(tileSizes);
}

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createLinalgTilingToParallelLoopsPass(ArrayRef<int64_t> tileSizes) {
  return std::make_unique<CIMGEMMTilingPass<loop::ParallelOp>>(tileSizes);
}

static PassRegistration<CIMGEMMTilingPass<loop::ForOp>>
    tiling_pass("cim-tile", "Tile operations in the cim dialect", [] {
      auto pass = std::make_unique<CIMGEMMTilingPass<loop::ForOp>>();
      pass->tileSizes.assign(clTileSizes.begin(), clTileSizes.end());
      return pass;
    });
