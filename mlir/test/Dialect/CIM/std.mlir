// RUN: mlir-opt %s -convert-cim-to-std | FileCheck %s

func @cim_func(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %devA = cim.memcpy_to_device(%arg0) : (memref<?x?xf32>) -> (memref<?x?xf32>)
  %devB = cim.memcpy_to_device(%arg1) : (memref<?x?xf32>) -> (memref<?x?xf32>)
  %devC = cim.memcpy_to_device(%arg2) : (memref<?x?xf32>) -> (memref<?x?xf32>)
  cim.matmul(%devA, %devB, %devC) : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  cim.memcpy(%devC, %arg2) { copyDirection = "toHost" } : memref<?x?xf32>, memref<?x?xf32>
  cim.dealloc %devA : memref<?x?xf32>
  cim.dealloc %devB : memref<?x?xf32>
  cim.dealloc %devC : memref<?x?xf32>
  return
}
// CHECK-LABEL: func @cim_func(
// CHECK-SAME: %[[arg0:.*]]: memref<?x?xf32>, %[[arg1:.*]]: memref<?x?xf32>, %[[arg2:.*]]: memref<?x?xf32>) {
//       CHECK: %[[devA:.*]] = call @cim_memcpy_to_device_viewsxsxf32_(%[[arg0]]) : (memref<?x?xf32>) -> memref<?x?xf32>
//       CHECK: %[[devB:.*]] = call @cim_memcpy_to_device_viewsxsxf32_(%[[arg1]]) : (memref<?x?xf32>) -> memref<?x?xf32>
//       CHECK: %[[devC:.*]] = call @cim_memcpy_to_device_viewsxsxf32_(%[[arg2]]) : (memref<?x?xf32>) -> memref<?x?xf32>
//       CHECK: call @cim_matmul_viewsxsxf32_viewsxsxf32_viewsxsxf32_(%[[devA]], %[[devB]], %[[devC]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
//       CHECK: call @cim_memcpy_viewsxsxf32_viewsxsxf32_toHost(%[[devC]], %[[arg2]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
//       CHECK: @cim_dealloc(%[[devA]]) : (memref<?x?xf32>) -> ()
//       CHECK: @cim_dealloc(%[[devB]]) : (memref<?x?xf32>) -> ()
//       CHECK: @cim_dealloc(%[[devC]]) : (memref<?x?xf32>) -> ()
//       CHECK: return
//       CHECK: }
//       CHECK: func @cim_memcpy_to_device_viewsxsxf32_(memref<?x?xf32>) -> memref<?x?xf32>
//       CHECK: func @cim_matmul_viewsxsxf32_viewsxsxf32_viewsxsxf32_(memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
//       CHECK: func @cim_memcpy_viewsxsxf32_viewsxsxf32_toHost(memref<?x?xf32>, memref<?x?xf32>)
//       CHECK: func @cim_dealloc(memref<?x?xf32>)
