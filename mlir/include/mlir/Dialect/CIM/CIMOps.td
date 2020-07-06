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
// TODO(adam-smnk): Fix error that occurs during library call name generation
// when different memrefs are used with the same functions, e.g.
// MemcpyOp called first with AnyMemRef and then with StridedMemRef
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
  let summary = "Writes a matrix into a CIM crossbar";
  let description = [{
    The "write_to_crossbar" operation writes the matrix into the specified
    crossbar.
    the provided vector and the matrix stored on a crossbar. The result is
    stored in the provided matrix.
    The shapes of the vector and matricies are expected to be valid.
    The tile ID specifies which crossbar executes the operation.

    Example:

    ```mlir
      // C(m, n) = A(m, k) * B(k, n)
      cim.write_to_crossbar(%tileId, %B) : i32, memref<1x?xi32>
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
  let summary = "general matrix-matrix multiplication";
  let description = [{
    The "gemm" operation performs matrix-matrix multiplication between
    the first matrix and the one stored on a crossbar. The result is
    stored in the second matrix.
    The shapes of the matricies are expected to be valid.
    The tile ID specifies which crossbar executes the operation.

    Example:

    ```mlir
      // C(m, n) = A(m, k) * B(k, n)
      cim.write_to_crossbar(%tileId, %B) : i32, memref<1x?xi32>
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
  let summary = "general vector-matrix multiplication";
  let description = [{
    The "gevm" operation performs vector-matrix multiplication between
    the first vector and the matrix stored on a crossbar. The result is
    stored in the second vector.
    The shapes of the vector and matricies are expected to be valid.
    The tile ID specifies which crossbar executes the operation.

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

def CIM_BarrierOp : CIM_Op<"barrier"> {
  let summary = "Waits until the specified CIM tile finishes computation";
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

def CIM_MatmulOp : CIM_Op<"matmul"> {
  let summary = "matrix multiplication";
  let description = [{
    The "matmul" operation performs matrix multiplication between
    the first two matricies. The result is stored in the third one.
    The shapes of the matricies are expected to be valid.

    Example:

    ```mlir
      // C(m, n) = A(m, k) * B(k, n)
      cim.matmul(%A, %B, %C) : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>
    ```
  }];

  let arguments = (ins AnyStridedMemRefOfRank<2>:$A,
                       AnyStridedMemRefOfRank<2>:$B,
                       AnyStridedMemRefOfRank<2>:$C);

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

def CIM_AllocOp : CIM_Op<"alloc"> {
  let summary = "allocate memory on CIM device";
  let description = [{
    The "alloc" operation allocates a region of memory, as specified by
    its memref type, on CIM device.
    The optional list of size operands are bound to the dynamic dimensions
    specified in its memref type.

    Example:

    ```mlir
      %0 = cim.alloc() : memref<2x3xi32>
    ```

    The optional list of dimension operands are bound to the dynamic dimensions
    specified in its memref type.

    ```mlir
      %0 = cim.alloc(%d1, %d2) : memref<?x?xi32>
    ```
  }];

  let arguments = (ins Variadic<Index>:$sizes);
  let results = (outs AnyMemRef);

  let assemblyFormat = "`(` ($sizes^)? `)` attr-dict `:` type(results)";
}

def CIM_DeallocOp : CIM_Op<"dealloc"> {
  let summary = "deallocate memory on CIM device";
  let description = [{
    The "dealloc" operation frees the region of CIM device memory referenced
    by a memref which was originally created by the "alloc" operation.

    Example:

    ```mlir
      %0 = cim.alloc(%d1, %d2) : memref<?x?xi32>
      cim.dealloc %0 : memref<?x?xi32>
    ```
  }];

  let arguments = (ins AnyMemRef:$memref);

  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def CIM_MemcpyOp : CIM_Op<"memcpy"> {
  let summary = "copy data between host and device";
  let description = [{
    The "memcpy" operation copies data between host and CIM device from
    the provided source to destination location.
    The memory references are expected to be compatible and have enough
    storage to perform the copy.

    Example:

    ```mlir
      cim.memcpy(%A, %devA) { copyDirection = "toDevice" } : memref<?x?xi32, offset: ?, strides: [?, 1]>, memref<?x?xi32>
      cim.memcpy(%devA, %A) { copyDirection = "toHost" } : memref<?x?xi32>, memref<?x?xi32, offset: ?, strides: [?, 1]>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$source,
    AnyMemRef:$destination,
    CIM_MemcpyOpDirectionAttr:$copyDirection);

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

def CIM_MemcpyToDeviceOp : CIM_Op<"memcpy_to_device"> {
  let summary = "copy data to CIM device";
  let description = [{
    The "memcpy_to_device" operation copies data from host to CIM device.
    The provided memory should be contiguous.

    Example:

    ```mlir
        %devA = cim.memcpy_to_device(%A) : (memref<?x?xi32>) -> (memref<?x?xi32>)
    ```
  }];

  let arguments = (ins AnyMemRef:$hostMemRef);
  let results = (outs AnyMemRef:$deviceMemRef);

  let assemblyFormat = "`(` operands `)` attr-dict `:` functional-type(operands, results)";
}

def CIM_MemcpyToHostOp : CIM_Op<"memcpy_to_host"> {
  let summary = "copy data from CIM device";
  let description = [{
    The "memcpy_to_host" operation copies data from CIM device to host.
    The provided memory should be contiguous.

    Example:

    ```mlir
      %devA = cim.alloc(%d1, %d2) : memref<?x?xi32>
      %A = cim.memcpy_to_host(%devA) : : (memref<?x?xi32>) -> (memref<?x?xi32>)
    ```
  }];

  let arguments = (ins AnyMemRef:$deviceMemRef);
  let results = (outs AnyMemRef:$hostMemRef);

  let assemblyFormat = "`(` operands `)` attr-dict `:` functional-type(operands, results)";
}

#endif // CIM_OPS
