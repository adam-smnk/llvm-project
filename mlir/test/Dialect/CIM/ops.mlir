// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// CHECK-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>


func @matmul_directional_memcpy(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c0 = constant 0 : index
  %A = view %arg0[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %B = view %arg0[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %C = view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %devA = cim.memcpy_to_device(%A) : (memref<?x?xf32, offset: ?, strides: [?, 1]>) -> (memref<?x?xf32, offset: ?, strides: [?, 1]>)
  %devB = cim.memcpy_to_device(%B) : (memref<?x?xf32, offset: ?, strides: [?, 1]>) -> (memref<?x?xf32, offset: ?, strides: [?, 1]>)
  %devC = cim.memcpy_to_device(%C) : (memref<?x?xf32, offset: ?, strides: [?, 1]>) -> (memref<?x?xf32, offset: ?, strides: [?, 1]>)
  cim.matmul(%devA, %devB, %devC) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  %matmulResult = cim.memcpy_to_host(%devC) : (memref<?x?xf32, offset: ?, strides: [?, 1]>) -> (memref<?x?xf32, offset: ?, strides: [?, 1]>)
  cim.dealloc %devA : memref<?x?xf32, offset: ?, strides: [?, 1]>
  cim.dealloc %devB : memref<?x?xf32, offset: ?, strides: [?, 1]>
  cim.dealloc %devC : memref<?x?xf32, offset: ?, strides: [?, 1]>
  linalg.copy(%matmulResult, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// CHECK-LABEL: func @matmul_directional_memcpy(%{{.*}}: memref<?xi8>,
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
//       CHECK: %[[matmulResult:.*]] = cim.memcpy_to_host(%[[devC]]) : (memref<?x?xf32, #[[strided2D]]>) -> memref<?x?xf32, #[[strided2D]]>
//       CHECK: cim.dealloc %[[devA]] : memref<?x?xf32, #[[strided2D]]>
//       CHECK: cim.dealloc %[[devB]] : memref<?x?xf32, #[[strided2D]]>
//       CHECK: cim.dealloc %[[devC]] : memref<?x?xf32, #[[strided2D]]>
//       CHECK: linalg.copy(%[[matmulResult]], %[[C]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>

func @matmul_generic_memcpy(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c0 = constant 0 : index
  %A = view %arg0[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %B = view %arg0[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %C = view %arg0[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %devA = cim.alloc(%M, %K) : memref<?x?xf32>
  %devB = cim.alloc(%K, %N) : memref<?x?xf32>
  %devC = cim.alloc(%M, %N) : memref<?x?xf32>
  cim.memcpy(%A, %devA) { copyDirection = "toDevice" } : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32>
  cim.memcpy(%B, %devB) { copyDirection = "toDevice" } : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32>
  cim.memcpy(%C, %devC) { copyDirection = "toDevice" } : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32>
  cim.matmul(%devA, %devB, %devC) : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  cim.memcpy(%devC, %C) { copyDirection = "toHost" } : memref<?x?xf32>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  cim.dealloc %devA : memref<?x?xf32>
  cim.dealloc %devB : memref<?x?xf32>
  cim.dealloc %devC : memref<?x?xf32>
  return
}
// CHECK-LABEL: func @matmul_generic_memcpy(%{{.*}}: memref<?xi8>,
// CHECK-SAME: [[M:arg[0-9]+]]: index
// CHECK-SAME: [[N:arg[0-9]+]]: index
// CHECK-SAME: [[K:arg[0-9]+]]: index
//       CHECK: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[devA:.*]] = cim.alloc(%{{.*}} : memref<?x?xf32>
//       CHECK: %[[devB:.*]] = cim.alloc(%{{.*}} : memref<?x?xf32>
//       CHECK: %[[devC:.*]] = cim.alloc(%{{.*}} : memref<?x?xf32>
//       CHECK: cim.memcpy(%[[A]], %[[devA]]) {copyDirection = "toDevice"} : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32>
//       CHECK: cim.memcpy(%[[B]], %[[devB]]) {copyDirection = "toDevice"} : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32>
//       CHECK: cim.memcpy(%[[C]], %[[devC]]) {copyDirection = "toDevice"} : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32>
//       CHECK: cim.matmul(%[[devA]], %[[devB]], %[[devC]]) : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
//       CHECK: cim.memcpy(%[[devC]], %[[C]]) {copyDirection = "toHost"} : memref<?x?xf32>, memref<?x?xf32, #[[strided2D]]>
//       CHECK: cim.dealloc %[[devA]] : memref<?x?xf32>
//       CHECK: cim.dealloc %[[devB]] : memref<?x?xf32>
//       CHECK: cim.dealloc %[[devC]] : memref<?x?xf32>
