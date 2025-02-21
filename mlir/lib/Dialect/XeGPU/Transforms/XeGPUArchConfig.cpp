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
  return;
}
