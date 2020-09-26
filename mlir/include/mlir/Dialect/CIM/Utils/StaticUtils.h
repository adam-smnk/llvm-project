#ifndef MLIR_DIALECT_CIM_UTILS_H_
#define MLIR_DIALECT_CIM_UTILS_H_

#include "mlir/Dialect/CIM/IR/CIMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include <algorithm>
#include <set>

namespace mlir {
namespace cim {

struct TransposeAnalysisResults {
  bool transposeA;
  bool transposeB;
  bool transposeResult;

  TransposeAnalysisResults() : TransposeAnalysisResults(false, false, false){};
  TransposeAnalysisResults(bool transposeA_, bool transposeB_,
                           bool transposeResult_)
      : transposeA(transposeA_), transposeB(transposeB_),
        transposeResult(transposeResult_){};
};

bool isAffineDimOnly(const ArrayRef<AffineExpr> &affineDims);

SmallVector<AffineMap, 8U> getResultMaps(ArrayAttr affineMaps);

SmallVector<AffineMap, 8U> getResultMaps(linalg::GenericOp genericOp);

SmallVector<ArrayRef<AffineExpr>, 8U> getResultDims(ArrayAttr affineMaps);

SmallVector<ArrayRef<AffineExpr>, 8U>
getResultDims(linalg::GenericOp genericOp);

std::vector<unsigned> getDimsPositions(const ArrayRef<AffineExpr> &affineDims);

AffineMap combineMaps(const ArrayRef<AffineMap> &affineMaps);

AffineMap combineMaps(const ArrayAttr &affineMaps);

SmallVector<AffineExpr, 8U> getPermutation(const AffineMap &originalMap,
                                           const AffineMap &targetMap,
                                           MLIRContext *context);

TransposeAnalysisResults
checkGEMMTransposes(const std::vector<unsigned> &dimsPosA,
                    const std::vector<unsigned> &dimsPosB,
                    const std::vector<unsigned> &dimsPosC);

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

template <typename T>
std::set<T> contractionReductionDims(std::set<T> dimsA, std::set<T> dimsB) {
  return setIntersection<T>(dimsA, dimsB);
}

template <typename T>
std::set<T> contractionOutputDims(std::set<T> dimsA, std::set<T> dimsB) {
  auto contrDims = contractionReductionDims<T>(dimsA, dimsB);

  auto uncontrDimsA = setDifference<T>(dimsA, contrDims);
  auto uncontrDimsB = setDifference<T>(dimsB, contrDims);

  return setUnion<T>(uncontrDimsA, uncontrDimsB);
}

} // namespace cim
} // namespace mlir

#endif /* MLIR_DIALECT_CIM_UTILS_H_ */