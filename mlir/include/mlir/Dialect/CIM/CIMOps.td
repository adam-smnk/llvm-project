//===-- CIMOps.td - CIM dialect operation definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines operations of the CIM dialect.
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
// CIM Dialect Attributes.
//===----------------------------------------------------------------------===//

def CIM_MemcpyOpToDevice : StrEnumAttrCase<"toDevice">;
def CIM_MemcpyOpToHost : StrEnumAttrCase<"toHost">;

def CIM_MemcpyOpDirectionAttr : StrEnumAttr<"MemcpyOpDirectionAttr",
    "defines direction of memory copy",
    [
      CIM_MemcpyOpToDevice,
      CIM_MemcpyOpToHost,
    ]>;

//===----------------------------------------------------------------------===//
// CIM Dialect operations.
//===----------------------------------------------------------------------===//

// Base class for CIM dialect ops.
class CIM_Op<string mnemonic, list<OpTrait> traits = []>
  : Op<CIM_Dialect, mnemonic, traits> {
  code libraryCallName = [{
    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  }];

  let extraClassDeclaration = libraryCallName;
}

def CIM_WriteToCrossbarOp : CIM_Op<"write_to_crossbar"> {
  let summary = "Write a matrix into a CIM crossbar";
  let description = [{
    The "write_to_crossbar" operation writes the provided matrix into
    the crossbar specified by its tile ID.
    The shapes of the matrix is expected to fit into the crossbar.
    The operation expects contiguous memref.

    Example:

    ```mlir
      // C(m, n) = A(m, k) * B(k, n)
      cim.write_to_crossbar(%tileId, %B) : i32, memref<?x?xi32>
    ```
  }];

  code libraryCallName = [{
    std::string getLibraryCallName() {
      std::string name = "cim_write_to_crossbar";

      llvm::raw_string_ostream ss(name);
      appendOperandPrecision(ss, B().getType());

      return ss.str();
    }
  }];

  let arguments = (ins SignlessIntegerLike:$tileId,
                       AnyStridedMemRefOfRank<2>:$B);

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

def CIM_GemmOp : CIM_Op<"gemm"> {
  let summary = "Perform general matrix-matrix multiplication on CIM";
  let description = [{
    The "gemm" operation performs matrix-matrix multiplication between
    the first matrix and the one stored on a crossbar. The result is
    stored in the second matrix.
    The tile ID specifies which crossbar executes the operation.
    The shapes of the matricies are expected to be valid.
    The operation expects contiguous memrefs.

    Example:

    ```mlir
      // C(m, n) = A(m, k) * B(k, n)
      cim.write_to_crossbar(%tileId, %B) : i32, memref<?x?xi32>
      cim.gemm(%tileId, %A, %C) : i32, memref<?x?xi32>, memref<?x?xi32>
    ```
  }];

  code libraryCallName = [{
    std::string getLibraryCallName() {
      std::string name = "cim_gemm";

      llvm::raw_string_ostream ss(name);
      appendOperandPrecision(ss, C().getType());

      return ss.str();
    }
  }];

  let arguments = (ins SignlessIntegerLike:$tileId,
                       AnyStridedMemRefOfRank<2>:$A,
                       AnyStridedMemRefOfRank<2>:$C);

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

def CIM_GevmOp : CIM_Op<"gevm"> {
  let summary = "Perform general vector-matrix multiplication on CIM";
  let description = [{
    The "gevm" operation performs vector-matrix multiplication between
    the first vector and the matrix stored on a crossbar. The result is
    stored in the second vector.
    The tile ID specifies which crossbar executes the operation.
    The shapes of the vector and matricies are expected to be valid.
    The operation expects contiguous memrefs.

    Example:

    ```mlir
      // C(n) = A(k) * B(k, n)
      cim.write_to_crossbar(%tileId, %B) : i32, memref<?x?xi32>
      cim.gevm(%tileId, %A, %C) : i32, memref<?xi32>, memref<?xi32>
    ```
  }];

  code libraryCallName = [{
    std::string getLibraryCallName() {
      std::string name = "cim_gevm";

      llvm::raw_string_ostream ss(name);
      appendOperandPrecision(ss, C().getType());

      return ss.str();
    }
  }];

  let arguments = (ins SignlessIntegerLike:$tileId,
                       AnyStridedMemRefOfRank<1>:$A,
                       AnyStridedMemRefOfRank<1>:$C);

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

def CIM_ContractionOp : CIM_Op<"contraction"> {
  let summary = "Represent tensor contraction";
  let description = [{
    The "contraction" operation is a placeholder for general tensor
    contraction computation.
    At the moment, contraction has no implementation for further
    lowering.

    Example:

    ```mlir
      // C(a,b,c) = A(a,e,f,g) * B(e,f,g,b,c)
      cim.contraction(%tileId, %A, %B, %C)
          : i32, memref<?x?x?x?xi32>, memref<?x?x?x?x?xi32>, memref<?x?x?xi32>
    ```
  }];

  code libraryCallName = [{
    std::string getLibraryCallName() {
      return "";
    }
  }];

  let arguments = (ins SignlessIntegerLike:$tileId,
                       AnyStridedMemRef:$A,
                       AnyStridedMemRef:$B,
                       AnyStridedMemRef:$C);

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

def CIM_BarrierOp : CIM_Op<"barrier"> {
  let summary = "Wait until the specified CIM tile finishes computation";
  let description = [{
    The "barrier" operation blocks program execution until computation on
    the specified tile is finished.

    Example:

    ```mlir
      // C(m, n) = A(m, k) * B(k, n)
      cim.write_to_crossbar(%tileId, %B) : i32, memref<1x?xi32>
      cim.gemm(%tileId, %A, %C) : i32, memref<?x?xi32>, memref<?x?xi32>

      // Wait until computation is finished
      cim.barrier(%tileId) : i32
    ```
  }];

  code libraryCallName = [{
    std::string getLibraryCallName() {
      return "cim_barrier";
    }
  }];

  let arguments = (ins SignlessIntegerLike:$tileId);

  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

#endif // CIM_OPS
