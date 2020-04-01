//===- CIMDialect.cpp - Implementation of the CIM operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIM related dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CIM/CIMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::cim;

//===----------------------------------------------------------------------===//
// CIMDialect
//===----------------------------------------------------------------------===//

CIMDialect::CIMDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/CIM/CIMOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cim {
#define GET_OP_CLASSES
#include "mlir/Dialect/CIM/CIMOps.cpp.inc"
} // namespace cim
} // namespace mlir
