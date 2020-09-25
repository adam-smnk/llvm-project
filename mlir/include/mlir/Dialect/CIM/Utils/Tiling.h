#ifndef MLIR_DIALECT__CIM__UTILS__TILING_H_
#define MLIR_DIALECT__CIM__UTILS__TILING_H_

#include "mlir/Dialect/CIM/IR/CIMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace cim {

void createCIMTiledGEMM(Operation *op, PatternRewriter &rewriter,
                        ConstantOp &tileId, const Value &matA,
                        const Value &matB, const Value &matC, uint32_t tileSize,
                        bool minWrites, unsigned numTilesOption);

} // namespace cim
} // namespace mlir

#endif /* MLIR_DIALECT__CIM__UTILS__TILING_H_ */