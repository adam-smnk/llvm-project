//===- MathTransformOps.cpp - Implementation of Math transform ops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Math/TransformOps/MathTransformOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyPolynomialApproximationPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  if (getOps().empty()) {
    populateMathPolynomialApproximationPatterns(patterns);
    return;
  }
  SmallVector<std::string> names;
  for (StringRef mnemonic : getOps().getAsValueRange<StringAttr>())
    names.push_back((math::MathDialect::getDialectNamespace() + "." + mnemonic)
                        .str());
  auto predicate = [&](StringRef name) { return llvm::is_contained(names, name); };
  populateMathF32ExpansionPatterns(patterns, predicate);
  populateMathPolynomialApproximationPatterns(patterns, predicate);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class MathTransformDialectExtension
    : public transform::TransformDialectExtension<
          MathTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MathTransformDialectExtension)

  MathTransformDialectExtension() {
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<math::MathDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Math/TransformOps/MathTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Math/TransformOps/MathTransformOps.cpp.inc"

void mlir::math::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<MathTransformDialectExtension>();
}
