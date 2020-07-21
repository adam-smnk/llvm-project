// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s


func @cim_gemm_32(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>) {
  %c0_i32 = constant 0 : i32
  cim.write_to_crossbar(%c0_i32, %arg1) : i32, memref<?x?xi32>
  cim.gemm(%c0_i32, %arg0, %arg2) : i32, memref<?x?xi32>, memref<?x?xi32>
  cim.barrier %c0_i32 : i32
  return
}
// CHECK-LABEL: func @cim_gemm_32(
// CHECK-SAME: %[[arg0:.*]]: memref<?x?xi32>, %[[arg1:.*]]: memref<?x?xi32>, %[[arg2:.*]]: memref<?x?xi32>) {
//       CHECK: %[[tileId:.*]] = constant 0 : i32
//       CHECK: cim.write_to_crossbar(%[[tileId]], %[[arg1]]) : i32, memref<?x?xi32>
//       CHECK: cim.gemm(%[[tileId]], %[[arg0]], %[[arg2]]) : i32, memref<?x?xi32>, memref<?x?xi32>
//       CHECK: cim.barrier %[[tileId]] : i32

func @cim_gevm_32(%arg0: memref<?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?xi32>) {
  %c0_i32 = constant 0 : i32
  cim.write_to_crossbar(%c0_i32, %arg1) : i32, memref<?x?xi32>
  cim.gevm(%c0_i32, %arg0, %arg2) : i32, memref<?xi32>, memref<?xi32>
  cim.barrier %c0_i32 : i32
  return
}
// CHECK-LABEL: func @cim_gevm_32(
// CHECK-SAME: %[[arg0:.*]]: memref<?xi32>, %[[arg1:.*]]: memref<?x?xi32>, %[[arg2:.*]]: memref<?xi32>) {
//       CHECK: %[[tileId:.*]] = constant 0 : i32
//       CHECK: cim.write_to_crossbar(%[[tileId]], %[[arg1]]) : i32, memref<?x?xi32>
//       CHECK: cim.gevm(%[[tileId]], %[[arg0]], %[[arg2]]) : i32, memref<?xi32>, memref<?xi32>
//       CHECK: cim.barrier %[[tileId]] : i32
