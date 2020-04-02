//===- LinalgToCIMPass.h - Convert Linalg to CIM dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_LINALGTOCIM_LINALGTOCIMPASS_H_
#define MLIR_CONVERSION_LINALGTOCIM_LINALGTOCIMPASS_H_

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class OwningRewritePatternList;

class MLIRContext;
class ModuleOp;
template <typename T> class OpPassBase;

/// Populate the given list with patterns that convert from Linalg to CIM.
void populateLinalgToCIMConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx);

/// Create a pass to convert Linalg operations to the CIM dialect.
std::unique_ptr<OpPassBase<ModuleOp>> createConvertLinalgToCIMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOCIM_LINALGTOCIMPASS_H_
