//===-- Passes.td - XeGPU transformation definition file ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def XeGPUFoldAliasOps : Pass<"xegpu-fold-alias-ops"> {
  let summary = "Fold alias ops into XeGPU ops";
  let description = [{
    The pass folds aliasing ops into XeGPU ops that they operate on the original
    source references.
  }];
  let dependentDialects = [
      "memref::MemRefDialect", "xegpu::XeGPUDialect"
  ];
}

def XeGPUArchConfig : Pass<"xegpu-arch-config"> {
  let summary = "Attach an XeGPU target DLTI configuration";
  let description = [{
    Attaches DLTI descriptor with hardware characteristics for the selected
    target architecture.

    The entries indicate which features are available and their supported
    configuration variants if any.
  }];
  let options = [
    Option<"arch", "arch", "int",
            /*default=*/"1",
           "Target architecture">,
  ];
  let dependentDialects = [ "mlir::DLTIDialect" ];
}

#endif // MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD
