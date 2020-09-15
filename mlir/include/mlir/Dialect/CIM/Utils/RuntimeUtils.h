#ifndef MLIR_DIALECT_CIM_RUNTIME_UTILS_H_
#define MLIR_DIALECT_CIM_RUNTIME_UTILS_H_

#include "mlir/Dialect/CIM/IR/CIMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace cim {

Value transposeMemRef(Operation *op, PatternRewriter &rewriter,
                      const Value &memRef, const AffineMap &memRefMap,
                      const AffineMap &targetMap);

SmallVector<Value, 8U> getMemRefSizes(Operation *op, PatternRewriter &rewriter,
                                      const Value &memRef,
                                      const ArrayRef<unsigned> &dimsPositions);

Value calculateDimsSize(Operation *op, PatternRewriter &rewriter,
                        const ArrayRef<Value> &dimOperands);

SmallVector<Value, 8U> calculateDimsStrides(Operation *op,
                                            PatternRewriter &rewriter,
                                            const ArrayRef<Value> &dimOperands);

Value calculateLinearIndex(Operation *op, PatternRewriter &rewriter,
                           const ArrayRef<Value> &dimIterators,
                           const ArrayRef<Value> &dimStrides);

Value allocateDuplicate(Operation *op, PatternRewriter &rewriter,
                        const Value &memRef);

void elementwiseAddition(Operation *op, PatternRewriter &rewriter,
                         const Value &inputMemRef, const Value &outputMemRef);

Value groupDimensions(Operation *op, PatternRewriter &rewriter,
                      const Value &memRef, const AffineMap &memRefMap,
                      const ArrayAttr &reassociation);

void reshapeCopy(Operation *op, PatternRewriter &rewriter,
                 const Value &inputMemRef, const Value &outputMemRef,
                 const ArrayAttr &reassociation,
                 ArrayRef<unsigned> permutation = ArrayRef<unsigned>(),
                 bool performElementwiseSum = false);

Value createIndexConst(Operation *op, PatternRewriter &rewriter, int64_t value);

Value minSigned(Operation *op, PatternRewriter &rewriter, const Value &lhs,
                const Value &rhs);

Value maxSigned(Operation *op, PatternRewriter &rewriter, const Value &lhs,
                const Value &rhs);

Value minUnsigned(Operation *op, PatternRewriter &rewriter, const Value &lhs,
                  const Value &rhs);

Value maxUnsigned(Operation *op, PatternRewriter &rewriter, const Value &lhs,
                  const Value &rhs);

Value calculateNumTiles(Operation *op, PatternRewriter &rewriter,
                        const Value &tileSize, const Value &dimMaxSize);

Value allocateTile(Operation *op, PatternRewriter &rewriter, const Value &mat,
                   const Value &row, const Value &col, const Value &tileSize,
                   bool copyData = false);

void storeTile(Operation *op, PatternRewriter &rewriter, const Value &tile,
               const Value &mat, const Value &row, const Value &col,
               const Value &tileSize);

} // namespace cim
} // namespace mlir

#endif /* MLIR_DIALECT_CIM_RUNTIME_UTILS_H_ */