// RUN: mlir-opt %s -convert-cim-to-std | FileCheck %s

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
//       CHECK: call @cim_write_to_crossbar_i32(%[[tileId]], %[[arg1]]) : (i32, memref<?x?xi32>) -> ()
//       CHECK: call @cim_gemm_i32(%[[tileId]], %[[arg0]], %[[arg2]]) : (i32, memref<?x?xi32>, memref<?x?xi32>) -> ()
//       CHECK: call @cim_barrier(%[[tileId]]) : (i32) -> ()
//       CHECK: return
//       CHECK: }

func @cim_gemm_8(%arg0: memref<?x?xi8>, %arg1: memref<?x?xi8>, %arg2: memref<?x?xi8>) {
  %c0_i32 = constant 0 : i32
  cim.write_to_crossbar(%c0_i32, %arg1) : i32, memref<?x?xi8>
  cim.gemm(%c0_i32, %arg0, %arg2) : i32, memref<?x?xi8>, memref<?x?xi8>
  cim.barrier %c0_i32 : i32
  return
}
// CHECK-LABEL: func @cim_gemm_8(
// CHECK-SAME: %[[arg0:.*]]: memref<?x?xi8>, %[[arg1:.*]]: memref<?x?xi8>, %[[arg2:.*]]: memref<?x?xi8>) {
//       CHECK: %[[tileId:.*]] = constant 0 : i32
//       CHECK: call @cim_write_to_crossbar_i8(%[[tileId]], %[[arg1]]) : (i32, memref<?x?xi8>) -> ()
//       CHECK: call @cim_gemm_i8(%[[tileId]], %[[arg0]], %[[arg2]]) : (i32, memref<?x?xi8>, memref<?x?xi8>) -> ()
//       CHECK: call @cim_barrier(%[[tileId]]) : (i32) -> ()
//       CHECK: return
//       CHECK: }

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
//       CHECK: call @cim_write_to_crossbar_i32(%[[tileId]], %[[arg1]]) : (i32, memref<?x?xi32>) -> ()
//       CHECK: call @cim_gevm_i32(%[[tileId]], %[[arg0]], %[[arg2]]) : (i32, memref<?xi32>, memref<?xi32>) -> ()
//       CHECK: call @cim_barrier(%[[tileId]]) : (i32) -> ()
//       CHECK: return
//       CHECK: }

func @cim_gevm_8(%arg0: memref<?xi8>, %arg1: memref<?x?xi8>, %arg2: memref<?xi8>) {
  %c0_i32 = constant 0 : i32
  cim.write_to_crossbar(%c0_i32, %arg1) : i32, memref<?x?xi8>
  cim.gevm(%c0_i32, %arg0, %arg2) : i32, memref<?xi8>, memref<?xi8>
  cim.barrier %c0_i32 : i32
  return
}
// CHECK-LABEL: func @cim_gevm_8(
// CHECK-SAME: %[[arg0:.*]]: memref<?xi8>, %[[arg1:.*]]: memref<?x?xi8>, %[[arg2:.*]]: memref<?xi8>) {
//       CHECK: %[[tileId:.*]] = constant 0 : i32
//       CHECK: call @cim_write_to_crossbar_i8(%[[tileId]], %[[arg1]]) : (i32, memref<?x?xi8>) -> ()
//       CHECK: call @cim_gevm_i8(%[[tileId]], %[[arg0]], %[[arg2]]) : (i32, memref<?xi8>, memref<?xi8>) -> ()
//       CHECK: call @cim_barrier(%[[tileId]]) : (i32) -> ()
//       CHECK: return
//       CHECK: }

//       CHECK: func @cim_write_to_crossbar_i32(i32, memref<?x?xi32>)
//       CHECK: func @cim_gemm_i32(i32, memref<?x?xi32>, memref<?x?xi32>)
//       CHECK: func @cim_barrier(i32)
//       CHECK: func @cim_write_to_crossbar_i8(i32, memref<?x?xi8>)
//       CHECK: func @cim_gevm_i8(i32, memref<?xi8>, memref<?xi8>)