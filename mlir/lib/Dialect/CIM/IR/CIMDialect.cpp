//===- CIMDialect.cpp - Implementation of the CIM operations --------------===//
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

static void appendMangledType(llvm::raw_string_ostream &ss, Type t) {
  if (auto memref = t.dyn_cast<MemRefType>()) {
    ss << "view";
    for (auto size : memref.getShape())
      if (size < 0)
        ss << "sx";
      else
        ss << size << "x";
    appendMangledType(ss, memref.getElementType());
  } else if (t.isSignlessIntOrIndexOrFloat()) {
    ss << t;
  } else {
    llvm_unreachable("Invalid type for cim library name mangling");
  }
}

std::string mlir::cim::generateLibraryCallName(Operation *op) {
  std::string name(op->getName().getStringRef().str());
  name.reserve(128);
  std::replace(name.begin(), name.end(), '.', '_');
  llvm::raw_string_ostream ss(name);

  // Skip types from memory management operations
  if (!(isa<cim::AllocOp>(op) || isa<cim::DeallocOp>(op))) {
    ss << "_";
    auto types = op->getOperandTypes();
    interleave(
        types.begin(), types.end(), [&](Type t) { appendMangledType(ss, t); },
        [&]() { ss << "_"; });

    ss << "_";
    auto attrs = op->getAttrs();
    interleave(
        attrs.begin(), attrs.end(),
        [&](NamedAttribute attr) {
          ss << std::get<1>(attr).cast<StringAttr>().getValue();
        },
        [&]() { ss << "_"; });
  }

  return ss.str();
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
