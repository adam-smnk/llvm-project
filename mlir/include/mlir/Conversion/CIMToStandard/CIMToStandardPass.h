//===- CIMToStandardPass.h - Convert CIM to Standard dialect -----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_CIMTOSTANDARD_CIMTOSTANDARDPASS_H_
#define MLIR_CONVERSION_CIMTOSTANDARD_CIMTOSTANDARDPASS_H_

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class OwningRewritePatternList;

class MLIRContext;
class ModuleOp;
template <typename T>
class OpPassBase;

/// Populate the given list with patterns that convert from CIM to Standard.
void populateCIMToStandardConversionPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *ctx);

/// Create a pass to convert CIM operations to the Standard dialect.
std::unique_ptr<OpPassBase<ModuleOp>> createConvertCIMToStandardPass();

} // namespace mlir

#endif // MLIR_CONVERSION_CIMTOSTANDARD_CIMTOSTANDARDPASS_H_
