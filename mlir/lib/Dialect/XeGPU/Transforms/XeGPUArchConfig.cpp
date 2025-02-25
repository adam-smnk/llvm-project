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
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include <optional>

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

DataLayoutEntryAttr getEntry(OpBuilder builder, StringRef key, int64_t value) {
  return builder.getAttr<DataLayoutEntryAttr>(builder.getAttr<StringAttr>(key),
                                              builder.getI64IntegerAttr(value));
}

// Returns config entry if given `arch` has DPAS hardware.
std::optional<DataLayoutEntryAttr> getDpasConfig(OpBuilder builder,
                                                 xegpu::Arch arch) {
  SmallVector<DataLayoutEntryInterface> entries;
  entries.push_back(getEntry(builder, "repeat_count", 8));
  int64_t execSize = arch == xegpu::Arch::ARC ? 8 : 16;
  entries.push_back(getEntry(builder, "exec_size", execSize));
  entries.push_back(getEntry(builder, "depth", 8));

  auto cfgId = builder.getAttr<StringAttr>("DPAS_HW");
  auto dltiMap = builder.getAttr<MapAttr>(entries);

  return builder.getAttr<DataLayoutEntryAttr>(cfgId, dltiMap);
}

} // namespace

void XeGPUArchConfigPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  auto targetId =
      builder.getAttr<StringAttr>("gpu-intel-" + xegpu::stringifyArch(arch));
  FailureOr<Attribute> queryDevice =
      dlti::query(op, SmallVector<DataLayoutEntryKey>{targetId});
  // Do nothing if the device spec already exists.
  if (succeeded(queryDevice))
    return;

  SmallVector<DataLayoutEntryInterface> deviceEntries;
  std::optional<DataLayoutEntryAttr> dpasCfg = getDpasConfig(builder, arch);
  if (dpasCfg)
    deviceEntries.push_back(*dpasCfg);

  auto deviceSpecAttr = builder.getAttr<TargetDeviceSpecAttr>(deviceEntries);
  auto deviceAttr =
      builder.getAttr<DataLayoutEntryAttr>(targetId, deviceSpecAttr);

  // Update target system spec descriptor.
  // The device spec for the target arch is set or updated, if already
  // present. Other device specs are preserved.
  SmallVector<DataLayoutEntryInterface> systemEntries = {deviceAttr};
  if (auto systemAttr =
          op->getAttrOfType<TargetSystemSpecAttr>(TargetSystemSpecAttr::name)) {
    ArrayRef<DataLayoutEntryInterface> currentEntries = systemAttr.getEntries();
    systemEntries.append(currentEntries.begin(), currentEntries.end());
  }
  auto systemSpecAttr = builder.getAttr<TargetSystemSpecAttr>(
      SmallVector<DataLayoutEntryInterface>{systemEntries});
  op->setAttr(TargetSystemSpecAttr::name, systemSpecAttr);
}
