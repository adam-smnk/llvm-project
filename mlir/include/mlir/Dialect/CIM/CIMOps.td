//===-- CIMOps.td - CIM dialect operation definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines some operations of the CIM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIM_OPS
#define CIM_OPS

include "mlir/IR/OpBase.td"

def CIM_Dialect : Dialect {
  let name = "cim";
  let description = [{
    The 'cim' dialect defines types, operation and transformations
    that can be deployed to a Computation-in-Memory device.
  }];
}

//===----------------------------------------------------------------------===//
// CIM Dialect operations.
//===----------------------------------------------------------------------===//

// Base class for CIM dialect ops.
class CIM_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<CIM_Dialect, mnemonic, traits>;

def CIM_MatmulOp : CIM_Op<"matmul", []> {
  let summary = "matrix multiplication";
  let description = [{
    The "matmul" operation performs matrix multiplication between
    the first two matricies. The result is stored in the third one.
    The shapes of the matricies are expected to be valid.
  }];

  let arguments = (ins AnyStridedMemRefOfRank<2>:$A,
                       AnyStridedMemRefOfRank<2>:$B,
                       AnyStridedMemRefOfRank<2>:$C);

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

def CIM_MemcpyToDeviceOp : CIM_Op<"memcpy_to_device", [NoSideEffect]> {
  let summary = "copy data to CIM device";
  let description = [{
    The "memcpy_to_device" operation copies data from host to CIM device.
    The provided memory should be contiguous.
  }];

  let arguments = (ins AnyMemRef:$hostMemRef);
  let results = (outs AnyMemRef:$deviceMemRef);

  let assemblyFormat = "`(` operands `)` attr-dict `:` functional-type(operands, results)";
}

def CIM_MemcpyToHostOp : CIM_Op<"memcpy_to_host", [NoSideEffect]> {
  let summary = "copy data from CIM device";
  let description = [{
    The "memcpy_to_host" operation copies data from CIM device to host.
    The provided memory should be contiguous.
  }];

  let arguments = (ins AnyMemRef:$deviceMemRef);
  let results = (outs AnyMemRef:$hostMemRef);

  let assemblyFormat = "`(` operands `)` attr-dict `:` functional-type(operands, results)";
}

#endif // CIM_OPS
