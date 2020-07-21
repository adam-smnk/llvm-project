// RUN: mlir-opt %s -convert-linalg-to-cim | FileCheck %s

// CHECK-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[doubleTransposeInputPerm:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[doubleTransposeOutputPerm:.*]] = affine_map<(d0, d1) -> (d1, d0)>

func @matmul(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %A = view %arg0[%c0][%M, %K] : memref<?xi8> to memref<?x?xi8, offset: ?, strides: [?, 1]>
  %B = view %arg0[%c0][%K, %N] : memref<?xi8> to memref<?x?xi8, offset: ?, strides: [?, 1]>
  %C = view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xi8, offset: ?, strides: [?, 1]>
  linalg.matmul(%A, %B, %C) : memref<?x?xi8, offset: ?, strides: [?, 1]>, memref<?x?xi8, offset: ?, strides: [?, 1]>, memref<?x?xi8, offset: ?, strides: [?, 1]>
  return
}
// CHECK-LABEL: func @matmul(%{{.*}}: memref<?xi8>,
// CHECK-SAME: [[M:arg[0-9]+]]: index
// CHECK-SAME: [[N:arg[0-9]+]]: index
// CHECK-SAME: [[K:arg[0-9]+]]: index
//       CHECK: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xi8, #[[strided2D]]>
//       CHECK: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xi8, #[[strided2D]]>
//       CHECK: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xi8, #[[strided2D]]>
//       CHECK: %[[tileId:.*]] = constant 0 : i32
//       CHECK: cim.write_to_crossbar(%[[tileId]], %[[B]]) : i32, memref<?x?xi8, #[[strided2D]]>
//       CHECK: cim.gemm(%[[tileId]], %[[A]], %[[C]]) : i32, memref<?x?xi8, #[[strided2D]]>, memref<?x?xi8, #[[strided2D]]>
//       CHECK: cim.barrier %[[tileId]] : i32


#matmul_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d2, d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}
func @generic_matmul(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>) {
  linalg.generic #matmul_trait %arg0, %arg1, %arg2 {
  ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
    %3 = "std.muli"(%arg3, %arg4) : (i32, i32) -> i32
    %4 = "std.addi"(%3, %arg5) : (i32, i32) -> i32
    "linalg.yield"(%4) : (i32) -> ()
  } : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>
  return
}
// CHECK-LABEL: func @generic_matmul(
// CHECK-SAME: %[[arg0:.*]]: memref<?x?xi32>, %[[arg1:.*]]: memref<?x?xi32>, %[[arg2:.*]]: memref<?x?xi32>) {
//       CHECK: %[[tileId:.*]] = constant 0 : i32
//       CHECK: cim.write_to_crossbar(%[[tileId]], %[[arg1]]) : i32, memref<?x?xi32>
//       CHECK: cim.gemm(%[[tileId]], %[[arg0]], %[[arg2]]) : i32, memref<?x?xi32>, memref<?x?xi32>
//       CHECK: cim.barrier %[[tileId]] : i32


#map0_contraction_4x5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d4, d0, d1)>
#map1_contraction_4x5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d0, d1, d3, d5)>
#map2_contraction_4x5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d5)>

func @contraction_4x5(%arg0: memref<?x?x?x?xi32>, %arg1: memref<?x?x?x?x?xi32>, %arg2: memref<?x?x?xi32>) {
  linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0_contraction_4x5, #map1_contraction_4x5, #map2_contraction_4x5], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction", "parallel"]} %arg0, %arg1, %arg2 {
  ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
    %0 = muli %arg3, %arg4 : i32
    %1 = addi %0, %arg5 : i32
    linalg.yield %1 : i32
  }: memref<?x?x?x?xi32>, memref<?x?x?x?x?xi32>, memref<?x?x?xi32>
  return
}
// CHECK-LABEL: func @contraction_4x5(
// CHECK-SAME: %[[arg0:.*]]: memref<?x?x?x?xi32>, %[[arg1:.*]]: memref<?x?x?x?x?xi32>, %[[arg2:.*]]: memref<?x?x?xi32>) {
//       CHECK: %[[tileId:.*]] = constant 0 : i32
//       CHECK: cim.contraction(%[[tileId]], %[[arg0]], %[[arg1]], %[[arg2]]) : i32, memref<?x?x?x?xi32>, memref<?x?x?x?x?xi32>, memref<?x?x?xi32>
//       CHECK: cim.barrier %[[tileId]] : i32


#map0_double_transpose = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1_double_transpose = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2_double_transpose = affine_map<(d0, d1, d2) -> (d0, d1)>

func @gemm_double_transpose(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>) {
  linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0_double_transpose, #map1_double_transpose, #map2_double_transpose], iterator_types = ["parallel", "parallel", "reduction"]} %arg0, %arg1, %arg2 {
  ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):	// no predecessors
    %0 = muli %arg3, %arg4 : i32
    %1 = addi %0, %arg5 : i32
    linalg.yield %1 : i32
  }: memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>
  return
}
// CHECK-LABEL: func @gemm_double_transpose(
// CHECK-SAME: %[[arg0:.*]]: memref<?x?xi32>, %[[arg1:.*]]: memref<?x?xi32>, %[[arg2:.*]]: memref<?x?xi32>) {
//       CHECK: %[[tileId:.*]] = constant 0 : i32
//       CHECK: %[[Bdim0:.*]] = dim %arg1, 1 : memref<?x?xi32>
//       CHECK: %[[Bdim1:.*]] = dim %arg1, 0 : memref<?x?xi32>
//       CHECK: %[[Btranposed:.*]] = alloc(%[[Bdim0]], %[[Bdim1]]) : memref<?x?xi32>
//       CHECK: linalg.copy(%[[arg1]], %[[Btranposed]]) {inputPermutation = #[[doubleTransposeInputPerm]], outputPermutation = #[[doubleTransposeOutputPerm]]} : memref<?x?xi32>, memref<?x?xi32>
//       CHECK: cim.write_to_crossbar(%[[tileId]], %[[Btranposed]]) : i32, memref<?x?xi32>
//       CHECK: %[[Adim0:.*]] = dim %arg0, 1 : memref<?x?xi32>
//       CHECK: %[[Adim1:.*]] = dim %arg0, 0 : memref<?x?xi32>
//       CHECK: %[[Atranposed:.*]] = alloc(%[[Adim0]], %[[Adim1]]) : memref<?x?xi32>
//       CHECK: linalg.copy(%[[arg0]], %[[Atranposed]]) {inputPermutation = #[[doubleTransposeInputPerm]], outputPermutation = #[[doubleTransposeOutputPerm]]} : memref<?x?xi32>, memref<?x?xi32>
//       CHECK: cim.gemm(%[[tileId]], %[[Atranposed]], %[[arg2]]) : i32, memref<?x?xi32>, memref<?x?xi32>
//       CHECK: cim.barrier %[[tileId]] : i32
//       CHECK: dealloc %[[Btranposed]] : memref<?x?xi32>
//       CHECK: dealloc %[[Atranposed]] : memref<?x?xi32>
