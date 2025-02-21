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
include "mlir/IR/EnumAttr.td"

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

def Arch : I32EnumAttr<"Arch", "Intel GPU architecture",
    [
      I32EnumAttrCase<"PVC", 0, "pvc">,
      I32EnumAttrCase<"ARC", 1, "arc">,
    ]>{
  let cppNamespace = "mlir::xegpu";
  let genSpecializedAttr = 0;
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
    Option<"arch", "arch", "mlir::xegpu::Arch",
           /*default=*/"mlir::xegpu::Arch::PVC",
           "Target architecture",
           [{::llvm::cl::values(
                 clEnumValN(mlir::xegpu::Arch::PVC,
                            "pvc", "Intel PVC GPU"),
                 clEnumValN(mlir::xegpu::Arch::ARC,
                            "arc",
                            "Intel ARC GPU")
           )}]>
  ];
  let dependentDialects = [ "mlir::DLTIDialect" ];
}

#endif // MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD
