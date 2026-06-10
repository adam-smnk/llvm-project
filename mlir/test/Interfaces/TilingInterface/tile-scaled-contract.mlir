// RUN: mlir-opt %s --transform-interpreter --cse -split-input-file -verify-diagnostics | FileCheck %s

// Tiling `linalg.scaled_contract` must respect the block-scaling scheme encoded
// in the scale indexing maps (operands at index 1 and 3). For every iteration
// dimension that is block-scaled (`d floordiv B`), the tile size along that
// dimension must divide or be divisible by the block factor `B`. Otherwise a
// tile would span a dynamic number of scale blocks, which cannot be expressed
// by the static scale indexing maps.

// Block scaling on the LHS scale (dims `m` and `k`) with tile sizes that are
// exact multiples of the block factors (m: 32, k: 128). The scale tile is
// addressed with a `floordiv` of the iteration variable.
func.func @tile_block_scale_multiple_of_block(
    %A: tensor<256x512xi8>, %scaleA: tensor<8x4xf8E8M0FNU>,
    %B: tensor<128x512xi8>, %scaleB: tensor<128xf8E8M0FNU>,
    %C: tensor<256x128xf32>) -> tensor<256x128xf32> {
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
    %1:4 = transform.structured.tile_using_for %0 tile_sizes [32, 16, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MAP_M:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
//   CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0) -> (d0 floordiv 128)>
// CHECK-LABEL: func.func @tile_block_scale_multiple_of_block(
//  CHECK-SAME:     %[[A:[a-zA-Z0-9]+]]: tensor<256x512xi8>
//  CHECK-SAME:     %[[SCALE_A:[a-zA-Z0-9]+]]: tensor<8x4xf8E8M0FNU>
//  CHECK-SAME:     %[[B:[a-zA-Z0-9]+]]: tensor<128x512xi8>
//  CHECK-SAME:     %[[SCALE_B:[a-zA-Z0-9]+]]: tensor<128xf8E8M0FNU>
//  CHECK-SAME:     %[[C:[a-zA-Z0-9]+]]: tensor<256x128xf32>
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//       CHECK:   scf.for %[[IV_M:[a-zA-Z0-9]+]] = %{{.+}} to %{{.+}} step %[[C32]] iter_args(%[[INIT_M:.+]] = %[[C]])
//       CHECK:     scf.for %[[IV_N:[a-zA-Z0-9]+]] = %{{.+}} to %{{.+}} step %[[C16]] iter_args(%[[INIT_N:.+]] = %[[INIT_M]])
//       CHECK:       scf.for %[[IV_K:[a-zA-Z0-9]+]] = %{{.+}} to %{{.+}} step %[[C128]] iter_args(%[[INIT_K:.+]] = %[[INIT_N]])
//   CHECK-DAG:         %[[OFF_M:.+]] = affine.apply #[[$MAP_M]](%[[IV_M]])
//   CHECK-DAG:         %[[OFF_K:.+]] = affine.apply #[[$MAP_K]](%[[IV_K]])
//   CHECK-DAG:         %[[A_TILE:.+]] = tensor.extract_slice %[[A]][%[[IV_M]], %[[IV_K]]] [32, 128] [1, 1]
//   CHECK-DAG:         %[[SA_TILE:.+]] = tensor.extract_slice %[[SCALE_A]][%[[OFF_M]], %[[OFF_K]]] [1, 1] [1, 1]
//   CHECK-DAG:         %[[B_TILE:.+]] = tensor.extract_slice %[[B]][%[[IV_N]], %[[IV_K]]] [16, 128] [1, 1]
//   CHECK-DAG:         %[[SB_TILE:.+]] = tensor.extract_slice %[[SCALE_B]][%[[IV_N]]] [16] [1]
//   CHECK-DAG:         %[[C_TILE:.+]] = tensor.extract_slice %[[INIT_K]][%[[IV_M]], %[[IV_N]]] [32, 16] [1, 1]
//       CHECK:         %[[RES:.+]] = linalg.scaled_contract
//  CHECK-SAME:             ins(%[[A_TILE]], %[[SA_TILE]], %[[B_TILE]], %[[SB_TILE]] :
//  CHECK-SAME:             outs(%[[C_TILE]] :
//  CHECK-SAME:             -> tensor<32x16xf32>
//       CHECK:         tensor.insert_slice %[[RES]] into %[[INIT_K]][%[[IV_M]], %[[IV_N]]] [32, 16] [1, 1]

// -----

// Sub-block tiling: the reduction tile (k: 64) is a divisor of the block factor
// (128). This is allowed because every tile stays within a single scale block,
// so the scale tile keeps a static size of 1 along that dimension.
func.func @tile_block_scale_divisor_of_block(
    %A: tensor<256x512xi8>, %scaleA: tensor<8x4xf8E8M0FNU>,
    %B: tensor<128x512xi8>, %scaleB: tensor<128xf8E8M0FNU>,
    %C: tensor<256x128xf32>) -> tensor<256x128xf32> {
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
    %1:4 = transform.structured.tile_using_for %0 tile_sizes [32, 16, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MAP_M:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
//   CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0) -> (d0 floordiv 128)>
// CHECK-LABEL: func.func @tile_block_scale_divisor_of_block(
//  CHECK-SAME:     %[[A:[a-zA-Z0-9]+]]: tensor<256x512xi8>
//  CHECK-SAME:     %[[SCALE_A:[a-zA-Z0-9]+]]: tensor<8x4xf8E8M0FNU>
//       CHECK:   scf.for %[[IV_M:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}}
//       CHECK:     scf.for %[[IV_N:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}}
//       CHECK:       scf.for %[[IV_K:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}}
//   CHECK-DAG:         %[[OFF_M:.+]] = affine.apply #[[$MAP_M]](%[[IV_M]])
//   CHECK-DAG:         %[[OFF_K:.+]] = affine.apply #[[$MAP_K]](%[[IV_K]])
//   CHECK-DAG:         %[[A_TILE:.+]] = tensor.extract_slice %[[A]][%[[IV_M]], %[[IV_K]]] [32, 64] [1, 1]
//   CHECK-DAG:         %[[SA_TILE:.+]] = tensor.extract_slice %[[SCALE_A]][%[[OFF_M]], %[[OFF_K]]] [1, 1] [1, 1]
//       CHECK:         %[[RES:.+]] = linalg.scaled_contract
//  CHECK-SAME:             -> tensor<32x16xf32>

// -----

// Scalar (whole-tensor) scale on the LHS and per-dimension scale on the RHS
// impose no block-scaling constraint, so the tile sizes can be arbitrary
// (and need not divide the operand shapes). The scalar scale is passed through
// untiled.
func.func @tile_scalar_and_dimension_scale(
    %A: tensor<256x100xi8>, %scaleA: tensor<f8E8M0FNU>,
    %B: tensor<100x128xi8>, %scaleB: tensor<128xf8E8M0FNU>,
    %C: tensor<256x128xf32>) -> tensor<256x128xf32> {
  %D = linalg.scaled_contract
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> ()>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %scaleA, %B, %scaleB
      : tensor<256x100xi8>, tensor<f8E8M0FNU>, tensor<100x128xi8>, tensor<128xf8E8M0FNU>)
    outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
  return %D : tensor<256x128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.tile_using_for %0 tile_sizes [30, 17, 25]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}
// CHECK-LABEL: func.func @tile_scalar_and_dimension_scale(
//  CHECK-SAME:     %[[A:[a-zA-Z0-9]+]]: tensor<256x100xi8>
//  CHECK-SAME:     %[[SCALE_A:[a-zA-Z0-9]+]]: tensor<f8E8M0FNU>
//       CHECK:   scf.for %[[IV_M:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}}
//       CHECK:     scf.for %[[IV_N:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}}
//       CHECK:       scf.for %[[IV_K:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}}
//       CHECK:         %[[RES:.+]] = linalg.scaled_contract
//  CHECK-SAME:             ins(%{{.+}}, %[[SCALE_A]], %{{.+}}, %{{.+}} :
//  CHECK-SAME:             -> tensor<?x?xf32>

// -----

// Dynamic operand shapes with static, valid tile sizes. The constraint is
// checked against the (constant) upper bound of the tile size, so block-scaled
// tiling of dynamic tensors is allowed as long as the tile sizes divide or are
// divisible by the block factors.
func.func @tile_dynamic_shapes(
    %A: tensor<?x?xi8>, %scaleA: tensor<?x?xf8E8M0FNU>,
    %B: tensor<?x?xi8>, %scaleB: tensor<?xf8E8M0FNU>,
    %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %D = linalg.scaled_contract
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (m floordiv 32, k floordiv 128)>,
      affine_map<(m, n, k) -> (n, k)>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %scaleA, %B, %scaleB
      : tensor<?x?xi8>, tensor<?x?xf8E8M0FNU>, tensor<?x?xi8>, tensor<?xf8E8M0FNU>)
    outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %D : tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.tile_using_for %0 tile_sizes [32, 16, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}
// CHECK-LABEL: func.func @tile_dynamic_shapes(
//       CHECK:   scf.for %[[IV_M:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}}
//       CHECK:     scf.for %[[IV_N:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}}
//       CHECK:       scf.for %[[IV_K:[a-zA-Z0-9]+]] = %{{.+}} step %{{.+}}
//       CHECK:         %[[RES:.+]] = linalg.scaled_contract
//  CHECK-SAME:             -> tensor<?x?xf32>

// -----

// Negative: the LHS scale block-scales dim `m` by 32, but `m` is tiled by 48,
// which neither divides nor is divisible by 32.
func.func @negative_tile_lhs_block_scale(
    %A: tensor<256x512xi8>, %scaleA: tensor<8x4xf8E8M0FNU>,
    %B: tensor<128x512xi8>, %scaleB: tensor<128xf8E8M0FNU>,
    %C: tensor<256x128xf32>) -> tensor<256x128xf32> {
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
    %1:4 = transform.structured.tile_using_for %0 tile_sizes [48, 16, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// Negative: the RHS scale block-scales dim `n` by 32, but `n` is tiled by 48,
// which neither divides nor is divisible by 32.
func.func @negative_tile_rhs_block_scale(
    %A: tensor<256x512xi8>, %scaleA: tensor<256xf8E8M0FNU>,
    %B: tensor<128x512xi8>, %scaleB: tensor<4x4xf8E8M0FNU>,
    %C: tensor<256x128xf32>) -> tensor<256x128xf32> {
  // expected-error @below {{'linalg.scaled_contract' op tile size 48 for dim 1 must divide or be divisible by scale factor 32}}
  // expected-error @below {{failed to tile operation}}
  // expected-error @below {{failed to generate tiling loops}}
  %D = linalg.scaled_contract
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (m)>,
      affine_map<(m, n, k) -> (n, k)>,
      affine_map<(m, n, k) -> (n floordiv 32, k floordiv 128)>,
      affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %scaleA, %B, %scaleB
      : tensor<256x512xi8>, tensor<256xf8E8M0FNU>, tensor<128x512xi8>, tensor<4x4xf8E8M0FNU>)
    outs(%C : tensor<256x128xf32>) -> tensor<256x128xf32>
  return %D : tensor<256x128xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.tile_using_for %0 tile_sizes [32, 48, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

// Negative: the reduction dim `k` is block-scaled by 128 (in both scales), but
// `k` is tiled by 48, which neither divides nor is divisible by 128.
func.func @negative_tile_reduction_block_scale(
    %A: tensor<256x512xi8>, %scaleA: tensor<8x4xf8E8M0FNU>,
    %B: tensor<128x512xi8>, %scaleB: tensor<128xf8E8M0FNU>,
    %C: tensor<256x128xf32>) -> tensor<256x128xf32> {
  // expected-error @below {{'linalg.scaled_contract' op tile size 48 for dim 2 must divide or be divisible by scale factor 128}}
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
    %1:4 = transform.structured.tile_using_for %0 tile_sizes [32, 16, 48]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}
