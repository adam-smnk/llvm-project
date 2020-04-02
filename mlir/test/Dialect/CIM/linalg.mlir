// RUN: mlir-opt %s -convert-linalg-to-cim | FileCheck %s

// CHECK-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>


func @matmul(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %A = view %arg0[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %B = view %arg0[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %C = view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// CHECK-LABEL: func @matmul(%{{.*}}: memref<?xi8>,
// CHECK-SAME: [[M:arg[0-9]+]]: index
// CHECK-SAME: [[N:arg[0-9]+]]: index
// CHECK-SAME: [[K:arg[0-9]+]]: index
//       CHECK: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[devA:.*]] = cim.memcpy_to_device(%[[A]]) : (memref<?x?xf32, #[[strided2D]]>) -> memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[devB:.*]] = cim.memcpy_to_device(%[[B]]) : (memref<?x?xf32, #[[strided2D]]>) -> memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[devC:.*]] = cim.memcpy_to_device(%[[C]]) : (memref<?x?xf32, #[[strided2D]]>) -> memref<?x?xf32, #[[strided2D]]>
//       CHECK: cim.matmul(%[[devA]], %[[devB]], %[[devC]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>
//       CHECK: cim.memcpy(%[[devC]], %[[C]]) {copyDirection = "toHost"} : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>
