#include "mlir/Dialect/CIM/Utils/StaticUtils.h"

#include "mlir/Support/Functional.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::cim;

SmallVector<AffineMap, 8U> mlir::cim::getResultMaps(ArrayAttr affineMaps) {
  return functional::map([](AffineMapAttr a) { return a.getValue(); },
                         affineMaps.getAsRange<AffineMapAttr>());
}

SmallVector<AffineMap, 8U>
mlir::cim::getResultMaps(linalg::GenericOp genericOp) {
  return getResultMaps(genericOp.indexing_maps());
}

SmallVector<ArrayRef<AffineExpr>, 8U>
mlir::cim::getResultDims(ArrayAttr affineMaps) {
  return functional::map(
      [](AffineMapAttr a) { return a.getValue().getResults(); },
      affineMaps.getAsRange<AffineMapAttr>());
}

SmallVector<ArrayRef<AffineExpr>, 8U>
mlir::cim::getResultDims(linalg::GenericOp genericOp) {
  return getResultDims(genericOp.indexing_maps());
}

std::vector<unsigned>
mlir::cim::getDimsPositions(const ArrayRef<AffineExpr> &affineDims) {
  std::vector<unsigned> dims;

  for (const auto &dim : affineDims) {
    if (dim.getKind() == AffineExprKind::DimId) {
      dims.push_back(dim.cast<AffineDimExpr>().getPosition());
    }
  }

  return dims;
}

AffineMap mlir::cim::combineMaps(const ArrayRef<AffineMap> &affineMaps) {
  SmallVector<AffineExpr, 8U> combinedDims;
  for (const auto &map : affineMaps) {
    auto dims = map.getResults();
    combinedDims.insert(combinedDims.end(), dims.begin(), dims.end());
  }

  return AffineMap::get(combinedDims.size(), 0, combinedDims);
}

AffineMap mlir::cim::combineMaps(const ArrayAttr &affineMaps) {
  return combineMaps(getResultMaps(affineMaps));
}

SmallVector<AffineExpr, 8U>
mlir::cim::getPermutation(const AffineMap &originalMap,
                          const AffineMap &targetMap, MLIRContext *context) {
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

// TODO(adam-smnk) Split the need to transpose contracted and
// uncontracted dimensions to avoid unnecessary tranposition when
// contraction dimensions already match.
TransposeAnalysisResults
mlir::cim::checkGEMMTransposes(const std::vector<unsigned> &dimsPosA,
                               const std::vector<unsigned> &dimsPosB,
                               const std::vector<unsigned> &dimsPosC) {
  std::set<unsigned> dimsSetA(dimsPosA.begin(), dimsPosA.end());
  std::set<unsigned> dimsSetB(dimsPosB.begin(), dimsPosB.end());
  std::set<unsigned> dimsSetC(dimsPosC.begin(), dimsPosC.end());

  auto contractionDims = contractionReductionDims<unsigned>(dimsSetA, dimsSetB);
  auto uncontrDimsA = setDifference<unsigned>(dimsSetA, contractionDims);
  auto uncontrDimsB = setDifference<unsigned>(dimsSetB, contractionDims);

  TransposeAnalysisResults result(false, false, false);

  // check if A and B dimensions match
  for (unsigned i = 0; i < contractionDims.size(); ++i) {
    unsigned aPos = uncontrDimsA.size() + i;

    if (dimsPosA[aPos] != dimsPosB[i]) {
      result.transposeA = true;
      result.transposeB = true;
      break;
    }
  }

  // check if A*B result matches C
  std::vector<unsigned> outputDimsPos;
  for (const auto &pos : uncontrDimsA) {
    outputDimsPos.push_back(pos);
  }
  for (const auto &pos : uncontrDimsB) {
    outputDimsPos.push_back(pos);
  }

  if (dimsPosC != outputDimsPos) {
    result.transposeResult = true;
  }

  return result;
}
