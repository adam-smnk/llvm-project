//===- CIMDialect.h - MLIR Dialect for CIM --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CIM related operations and puts them in the
// corresponding dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CIM_CIMDIALECT_H
#define MLIR_DIALECT_CIM_CIMDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace cim {

class CIMDialect : public Dialect {
public:
  CIMDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "cim"; }
};

/// Returns the name mangled library call name to disambiguate between different
/// overloads at the C level. The name mangling scheme is basic and uses MLIR
/// type names:
///   1. form a string which is the concatenation of the cim op name with all
///      the operand type names, separate by underscores;
///   2. drop the `cim.` prefix, and the `<`, `>`, `?` symbols from the type.
std::string generateLibraryCallName(Operation *op);

#define GET_OP_CLASSES
#include "mlir/Dialect/CIM/CIMOps.h.inc"

} // end namespace cim
} // end namespace mlir

#endif // MLIR_DIALECT_CIM_CIMDIALECT_H
