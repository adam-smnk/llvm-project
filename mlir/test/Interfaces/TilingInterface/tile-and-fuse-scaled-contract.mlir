// RUN: mlir-opt %s --transform-interpreter --cse -split-input-file -verify-diagnostics | FileCheck %s

// Tiling and fusing `linalg.scaled_contract` must respect the same block-scaling
// constraint as plain tiling: along every block-scaled iteration dimension the
// tile size must divide or be divisible by the block factor. These tests cover
// fusing a producer into the `scaled_contract` loop nest, fusing the
// `scaled_contract` into a consumer loop nest, and the corresponding negative
// cases where the constraint forbids the requested tiling.

// Fuse a `linalg.fill` producer (initializing the accumulator) into the tiled
// `scaled_contract` loop nest. The LHS scale is block-scaled on `m`/`k` and the
// tile sizes (m: 32, n: 16) are compatible with the block factors.
func.func @fuse_fill(%A: tensor<256x512xi8>, %scaleA: tensor<8x4xf8E8M0FNU>,
    %B: tensor<128x512xi8>, %scaleB: tensor<128xf8E8M0FNU>,
    %init: tensor<256x128xf32>) -> tensor<256x128xf32> {
  %cst = arith.constant 0.0 : f32
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<256x128xf32>) -> tensor<256x128xf32>
  %D = linalg.scaled_contract
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (m floordiv 32, k floordiv 128)>,
      affine_map<(m, n, k) -> (n, k)>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %scaleA, %B, %scaleB
      : tensor<256x512xi8>, tensor<8x4xf8E8M0FNU>, tensor<128x512xi8>, tensor<128xf8E8M0FNU>)
    outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
  return %D : tensor<256x128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [32, 16, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MAP_M:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-LABEL: func.func @fuse_fill(
//  CHECK-SAME:     %[[A:[a-zA-Z0-9]+]]: tensor<256x512xi8>
//  CHECK-SAME:     %[[SCALE_A:[a-zA-Z0-9]+]]: tensor<8x4xf8E8M0FNU>
//  CHECK-SAME:     %[[B:[a-zA-Z0-9]+]]: tensor<128x512xi8>
//  CHECK-SAME:     %[[SCALE_B:[a-zA-Z0-9]+]]: tensor<128xf8E8M0FNU>
//  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<256x128xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.0{{.*}} : f32
//       CHECK:   scf.for %[[IV_M:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}} iter_args(%[[INIT_M:.+]] = %[[INIT]])
//       CHECK:     scf.for %[[IV_N:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}} iter_args(%[[INIT_N:.+]] = %[[INIT_M]])
//   CHECK-DAG:       %[[OFF_M:.+]] = affine.apply #[[$MAP_M]](%[[IV_M]])
//   CHECK-DAG:       %[[A_TILE:.+]] = tensor.extract_slice %[[A]][%[[IV_M]], 0] [32, 512] [1, 1]
//   CHECK-DAG:       %[[SA_TILE:.+]] = tensor.extract_slice %[[SCALE_A]][%[[OFF_M]], 0] [1, 4] [1, 1]
//   CHECK-DAG:       %[[B_TILE:.+]] = tensor.extract_slice %[[B]][%[[IV_N]], 0] [16, 512] [1, 1]
//   CHECK-DAG:       %[[SB_TILE:.+]] = tensor.extract_slice %[[SCALE_B]][%[[IV_N]]] [16] [1]
//   CHECK-DAG:       %[[C_TILE:.+]] = tensor.extract_slice %[[INIT_N]][%[[IV_M]], %[[IV_N]]] [32, 16] [1, 1]
//       CHECK:       %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[C_TILE]] : tensor<32x16xf32>)
//       CHECK:       %[[RES:.+]] = linalg.scaled_contract
//  CHECK-SAME:           ins(%[[A_TILE]], %[[SA_TILE]], %[[B_TILE]], %[[SB_TILE]] :
//  CHECK-SAME:           outs(%[[FILL]] : tensor<32x16xf32>) -> tensor<32x16xf32>
//       CHECK:       tensor.insert_slice %[[RES]] into %[[INIT_N]][%[[IV_M]], %[[IV_N]]] [32, 16] [1, 1]

// -----

// Fuse the `scaled_contract` as a producer into a `linalg.copy` consumer loop
// nest. Tiling the consumer by (m: 32, n: 16) is compatible with the block
// scaling, so the `scaled_contract` is tiled and sunk into the loop.
func.func @fuse_into_copy(%A: tensor<256x512xi8>, %scaleA: tensor<8x4xf8E8M0FNU>,
    %B: tensor<128x512xi8>, %scaleB: tensor<128xf8E8M0FNU>,
    %C: tensor<256x128xf32>, %out: tensor<256x128xf32>) -> tensor<256x128xf32> {
  %D = linalg.scaled_contract
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (m floordiv 32, k floordiv 128)>,
      affine_map<(m, n, k) -> (n, k)>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %scaleA, %B, %scaleB
      : tensor<256x512xi8>, tensor<8x4xf8E8M0FNU>, tensor<128x512xi8>, tensor<128xf8E8M0FNU>)
    outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
  %E = linalg.copy ins(%D : tensor<256x128xf32>) outs(%out : tensor<256x128xf32>) -> tensor<256x128xf32>
  return %E : tensor<256x128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.copy"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [32, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MAP_M:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-LABEL: func.func @fuse_into_copy(
//  CHECK-SAME:     %[[A:[a-zA-Z0-9]+]]: tensor<256x512xi8>
//  CHECK-SAME:     %[[SCALE_A:[a-zA-Z0-9]+]]: tensor<8x4xf8E8M0FNU>
//  CHECK-SAME:     %[[B:[a-zA-Z0-9]+]]: tensor<128x512xi8>
//  CHECK-SAME:     %[[SCALE_B:[a-zA-Z0-9]+]]: tensor<128xf8E8M0FNU>
//  CHECK-SAME:     %[[C:[a-zA-Z0-9]+]]: tensor<256x128xf32>
//  CHECK-SAME:     %[[OUT:[a-zA-Z0-9]+]]: tensor<256x128xf32>
//       CHECK:   scf.for %[[IV_M:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}} iter_args(%[[INIT_M:.+]] = %[[OUT]])
//       CHECK:     scf.for %[[IV_N:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}} iter_args(%[[INIT_N:.+]] = %[[INIT_M]])
//   CHECK-DAG:       %[[OFF_M:.+]] = affine.apply #[[$MAP_M]](%[[IV_M]])
//   CHECK-DAG:       %[[A_TILE:.+]] = tensor.extract_slice %[[A]][%[[IV_M]], 0] [32, 512] [1, 1]
//   CHECK-DAG:       %[[SA_TILE:.+]] = tensor.extract_slice %[[SCALE_A]][%[[OFF_M]], 0] [1, 4] [1, 1]
//   CHECK-DAG:       %[[B_TILE:.+]] = tensor.extract_slice %[[B]][%[[IV_N]], 0] [16, 512] [1, 1]
//   CHECK-DAG:       %[[SB_TILE:.+]] = tensor.extract_slice %[[SCALE_B]][%[[IV_N]]] [16] [1]
//   CHECK-DAG:       %[[C_TILE:.+]] = tensor.extract_slice %[[C]][%[[IV_M]], %[[IV_N]]] [32, 16] [1, 1]
//       CHECK:       %[[SC:.+]] = linalg.scaled_contract
//  CHECK-SAME:           ins(%[[A_TILE]], %[[SA_TILE]], %[[B_TILE]], %[[SB_TILE]] :
//  CHECK-SAME:           outs(%[[C_TILE]] : tensor<32x16xf32>) -> tensor<32x16xf32>
//       CHECK:       %[[OUT_TILE:.+]] = tensor.extract_slice %[[INIT_N]][%[[IV_M]], %[[IV_N]]] [32, 16] [1, 1]
//       CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[SC]] : tensor<32x16xf32>) outs(%[[OUT_TILE]] : tensor<32x16xf32>)
//       CHECK:       tensor.insert_slice %[[COPY]] into %[[INIT_N]][%[[IV_M]], %[[IV_N]]] [32, 16] [1, 1]

// -----

// Negative: fusing a `fill` producer requires tiling the `scaled_contract` root,
// but `m` is block-scaled by 32 and tiled by 48. The root tiling fails, so the
// whole transform fails (no loops are generated).
func.func @negative_fuse_fill(%A: tensor<256x512xi8>, %scaleA: tensor<8x4xf8E8M0FNU>,
    %B: tensor<128x512xi8>, %scaleB: tensor<128xf8E8M0FNU>,
    %init: tensor<256x128xf32>) -> tensor<256x128xf32> {
  %cst = arith.constant 0.0 : f32
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<256x128xf32>) -> tensor<256x128xf32>
  // expected-error @below {{'linalg.scaled_contract' op tile size 48 for dim 0 must divide or be divisible by scale factor 32}}
  // expected-error @below {{failed to tile operation}}
  // expected-error @below {{failed to generate tiling loops}}
  %D = linalg.scaled_contract
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (m floordiv 32, k floordiv 128)>,
      affine_map<(m, n, k) -> (n, k)>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %scaleA, %B, %scaleB
      : tensor<256x512xi8>, tensor<8x4xf8E8M0FNU>, tensor<128x512xi8>, tensor<128xf8E8M0FNU>)
    outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
  return %D : tensor<256x128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [48, 16, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Negative: tiling the `copy` consumer by (m: 48) is fine for the copy, but the
// `scaled_contract` producer is block-scaled on `m` by 32 and cannot be fused
// for a tile of 48. Fusion is skipped: the constraint error is reported and the
// `scaled_contract` stays outside the loop, computing the full result that the
// tiled copy then slices.
func.func @negative_fuse_into_copy(%A: tensor<256x512xi8>, %scaleA: tensor<8x4xf8E8M0FNU>,
    %B: tensor<128x512xi8>, %scaleB: tensor<128xf8E8M0FNU>,
    %C: tensor<256x128xf32>, %out: tensor<256x128xf32>) -> tensor<256x128xf32> {
  // expected-error @below {{'linalg.scaled_contract' op tile size 48 for dim 0 must divide or be divisible by scale factor 32}}
  %D = linalg.scaled_contract
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (m floordiv 32, k floordiv 128)>,
      affine_map<(m, n, k) -> (n, k)>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %scaleA, %B, %scaleB
      : tensor<256x512xi8>, tensor<8x4xf8E8M0FNU>, tensor<128x512xi8>, tensor<128xf8E8M0FNU>)
    outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
  %E = linalg.copy ins(%D : tensor<256x128xf32>) outs(%out : tensor<256x128xf32>) -> tensor<256x128xf32>
  return %E : tensor<256x128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.copy"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.fuse %0 tile_sizes [48, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func.func @negative_fuse_into_copy(
//  CHECK-SAME:     %[[A:[a-zA-Z0-9]+]]: tensor<256x512xi8>
//  CHECK-SAME:     %[[SCALE_A:[a-zA-Z0-9]+]]: tensor<8x4xf8E8M0FNU>
//  CHECK-SAME:     %[[B:[a-zA-Z0-9]+]]: tensor<128x512xi8>
//  CHECK-SAME:     %[[SCALE_B:[a-zA-Z0-9]+]]: tensor<128xf8E8M0FNU>
//  CHECK-SAME:     %[[C:[a-zA-Z0-9]+]]: tensor<256x128xf32>
//  CHECK-SAME:     %[[OUT:[a-zA-Z0-9]+]]: tensor<256x128xf32>
// The scaled_contract is NOT fused: it computes the full result outside the loop.
//       CHECK:   %[[FULL:.+]] = linalg.scaled_contract
//  CHECK-SAME:       outs(%[[C]] : tensor<256x128xf32>) -> tensor<256x128xf32>
//       CHECK:   scf.for %[[IV_M:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}}
//       CHECK:     scf.for %[[IV_N:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}}
//       CHECK:       %[[D_TILE:.+]] = tensor.extract_slice %[[FULL]][%[[IV_M]], %[[IV_N]]] [%{{.+}}, 16] [1, 1]
//       CHECK:       %[[COPY:.+]] = linalg.copy ins(%[[D_TILE]] :
//       CHECK:       tensor.insert_slice %[[COPY]]
