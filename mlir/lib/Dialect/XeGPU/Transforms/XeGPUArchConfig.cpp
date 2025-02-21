//===- XeGPUArchConfig.cpp - XeGPU architecture config ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/PassesEnums.cpp.inc"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUARCHCONFIG
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

using namespace mlir;

namespace {

struct XeGPUArchConfigPass final
    : public xegpu::impl::XeGPUArchConfigBase<XeGPUArchConfigPass> {
  using Base::Base;

  void runOnOperation() override;
};

} // namespace

void XeGPUArchConfigPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(&getContext());

  auto targetId =
      StringAttr::get(ctx, "gpu-intel-" + xegpu::stringifyArch(arch));
  // Do nothing if the target system spec already exists.
  if (succeeded(dlti::query(op, SmallVector<DataLayoutEntryKey>{targetId})))
    return;

  llvm::errs() << *op << " - No attr yet\n";
  // auto targetAttr = builder.getAttr<DataLayoutEntryAttr>(
  //     StringAttr::get(ctx, targetId), Attribute());
  // // auto deviceSpecAttr = builder.getAttr<TargetDeviceSpecAttr>();
  // auto systemAttr = builder.getAttr<TargetSystemSpecAttr>(
  //     SmallVector<DataLayoutEntryInterface>{targetAttr});
  // op->setAttr(systemAttr.name, systemAttr);

  return;
}
