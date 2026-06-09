// RUN: mlir-opt %s --transform-interpreter --cse -split-input-file -verify-diagnostics | FileCheck %s

// Tile `linalg.mx_pack` along its scaling dimensions when the tile sizes are
// equal to the scale block sizes. The `scale_dest` result tile is inferred from
// the data tile: offset = floordiv(offset, block), size = ceildiv(size, block).
func.func @mx_pack_tile_block_aligned(%src: tensor<128x768xf16>,
    %data: tensor<128x768xi8>, %scales: tensor<16x24xf8E8M0FNU>)
    -> (tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>) {
  %0:2 = linalg.mx_pack %src
    scale_dims = [0, 1]
    scale_blocks = [8, 32]
    into %data, %scales
    : tensor<128x768xf16> -> tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
  return %0#0, %0#1 : tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.mx_pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-LABEL: func.func @mx_pack_tile_block_aligned(
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: tensor<128x768xf16>
//  CHECK-SAME:     %[[DATA:[a-zA-Z0-9]+]]: tensor<128x768xi8>
//  CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<16x24xf8E8M0FNU>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//   CHECK-DAG:   %[[C768:.+]] = arith.constant 768 : index
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//       CHECK:   %[[OUTER:.+]]:2 = scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[C128]] step %[[C8]]
//  CHECK-SAME:       iter_args(%[[ITER0_0:.+]] = %[[DATA]], %[[ITER0_1:.+]] = %[[SCALES]])
//       CHECK:     %[[INNER:.+]]:2 = scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[C768]] step %[[C32]]
//  CHECK-SAME:         iter_args(%[[ITER1_0:.+]] = %[[ITER0_0]], %[[ITER1_1:.+]] = %[[ITER0_1]])
//   CHECK-DAG:       %[[SRC_TILE:.+]] = tensor.extract_slice %[[SRC]][%[[IV0]], %[[IV1]]] [8, 32] [1, 1]
//   CHECK-DAG:       %[[DATA_TILE:.+]] = tensor.extract_slice %[[ITER1_0]][%[[IV0]], %[[IV1]]] [8, 32] [1, 1]
//   CHECK-DAG:       %[[OFF0:.+]] = affine.apply #[[$MAP0]](%[[IV0]])
//   CHECK-DAG:       %[[OFF1:.+]] = affine.apply #[[$MAP1]](%[[IV1]])
//       CHECK:       %[[SCALE_TILE:.+]] = tensor.extract_slice %[[ITER1_1]][%[[OFF0]], %[[OFF1]]] [1, 1] [1, 1]
//       CHECK:       %[[PACK_DATA:.+]], %[[PACK_SCALE:.+]] = linalg.mx_pack %[[SRC_TILE]]
//  CHECK-SAME:           scale_dims = [0, 1] scale_blocks = [8, 32]
//  CHECK-SAME:           into %[[DATA_TILE]], %[[SCALE_TILE]]
//       CHECK:       %[[INS_DATA:.+]] = tensor.insert_slice %[[PACK_DATA]] into %[[ITER1_0]][%[[IV0]], %[[IV1]]] [8, 32] [1, 1]
//       CHECK:       %[[INS_SCALE:.+]] = tensor.insert_slice %[[PACK_SCALE]] into %[[ITER1_1]][%[[OFF0]], %[[OFF1]]] [1, 1] [1, 1]
//       CHECK:       scf.yield %[[INS_DATA]], %[[INS_SCALE]]
//       CHECK:     scf.yield %[[INNER]]#0, %[[INNER]]#1
//       CHECK:   return %[[OUTER]]#0, %[[OUTER]]#1

// -----

// Tile `linalg.mx_pack` with tile sizes that are multiples (not equal) of the
// scale blocks. The inferred `scale_dest` tile has sizes ceildiv(tile, block).
func.func @mx_pack_tile_block_multiple(%src: tensor<128x768xf16>,
    %data: tensor<128x768xi8>, %scales: tensor<16x24xf8E8M0FNU>)
    -> (tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>) {
  %0:2 = linalg.mx_pack %src
    scale_dims = [0, 1]
    scale_blocks = [8, 32]
    into %data, %scales
    : tensor<128x768xf16> -> tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
  return %0#0, %0#1 : tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.mx_pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [16, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-LABEL: func.func @mx_pack_tile_block_multiple(
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: tensor<128x768xf16>
//  CHECK-SAME:     %[[DATA:[a-zA-Z0-9]+]]: tensor<128x768xi8>
//  CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<16x24xf8E8M0FNU>
//       CHECK:   %[[OUTER:.+]]:2 = scf.for %[[IV0:[a-zA-Z0-9]+]]
//       CHECK:     %[[INNER:.+]]:2 = scf.for %[[IV1:[a-zA-Z0-9]+]]
//   CHECK-DAG:       %[[SRC_TILE:.+]] = tensor.extract_slice %[[SRC]][%[[IV0]], %[[IV1]]] [16, 64] [1, 1]
//   CHECK-DAG:       %[[DATA_TILE:.+]] = tensor.extract_slice %{{.+}}[%[[IV0]], %[[IV1]]] [16, 64] [1, 1]
//   CHECK-DAG:       %[[OFF0:.+]] = affine.apply #[[$MAP0]](%[[IV0]])
//   CHECK-DAG:       %[[OFF1:.+]] = affine.apply #[[$MAP1]](%[[IV1]])
//       CHECK:       %[[SCALE_TILE:.+]] = tensor.extract_slice %{{.+}}[%[[OFF0]], %[[OFF1]]] [2, 2] [1, 1]
//       CHECK:       %[[PACK_DATA:.+]], %[[PACK_SCALE:.+]] = linalg.mx_pack %[[SRC_TILE]]
//  CHECK-SAME:           into %[[DATA_TILE]], %[[SCALE_TILE]]
//       CHECK:       tensor.insert_slice %[[PACK_DATA]] into %{{.+}}[%[[IV0]], %[[IV1]]] [16, 64] [1, 1]
//       CHECK:       tensor.insert_slice %[[PACK_SCALE]] into %{{.+}}[%[[OFF0]], %[[OFF1]]] [2, 2] [1, 1]

// -----

// Tile `linalg.mx_pack` whose `scale_dims` is a strict subset of the data dims.
// The non-scaling dim (0) is a reduction and is left untiled (tile size 0); only
// the scaling dim (1) is tiled, yielding a 1-D `scale_dest` tile.
func.func @mx_pack_tile_subset_scale_dims(%src: tensor<128x768xf16>,
    %data: tensor<128x768xi8>, %scales: tensor<24xf8E8M0FNU>)
    -> (tensor<128x768xi8>, tensor<24xf8E8M0FNU>) {
  %0:2 = linalg.mx_pack %src
    scale_dims = [1]
    scale_blocks = [32]
    into %data, %scales
    : tensor<128x768xf16> -> tensor<128x768xi8>, tensor<24xf8E8M0FNU>
  return %0#0, %0#1 : tensor<128x768xi8>, tensor<24xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.mx_pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-LABEL: func.func @mx_pack_tile_subset_scale_dims(
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: tensor<128x768xf16>
//  CHECK-SAME:     %[[DATA:[a-zA-Z0-9]+]]: tensor<128x768xi8>
//  CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<24xf8E8M0FNU>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C768:.+]] = arith.constant 768 : index
//   CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//       CHECK:   %[[LOOP:.+]]:2 = scf.for %[[IV:[a-zA-Z0-9]+]] = %[[C0]] to %[[C768]] step %[[C64]]
//  CHECK-SAME:       iter_args(%[[ITER0:.+]] = %[[DATA]], %[[ITER1:.+]] = %[[SCALES]])
//   CHECK-DAG:     %[[SRC_TILE:.+]] = tensor.extract_slice %[[SRC]][0, %[[IV]]] [128, 64] [1, 1]
//   CHECK-DAG:     %[[DATA_TILE:.+]] = tensor.extract_slice %[[ITER0]][0, %[[IV]]] [128, 64] [1, 1]
//   CHECK-DAG:     %[[OFF:.+]] = affine.apply #[[$MAP0]](%[[IV]])
//       CHECK:     %[[SCALE_TILE:.+]] = tensor.extract_slice %[[ITER1]][%[[OFF]]] [2] [1]
//       CHECK:     %[[PACK_DATA:.+]], %[[PACK_SCALE:.+]] = linalg.mx_pack %[[SRC_TILE]]
//  CHECK-SAME:         scale_dims = [1] scale_blocks = [32]
//  CHECK-SAME:         into %[[DATA_TILE]], %[[SCALE_TILE]]
//       CHECK:     %[[INS_DATA:.+]] = tensor.insert_slice %[[PACK_DATA]] into %[[ITER0]][0, %[[IV]]] [128, 64] [1, 1]
//       CHECK:     %[[INS_SCALE:.+]] = tensor.insert_slice %[[PACK_SCALE]] into %[[ITER1]][%[[OFF]]] [2] [1]
//       CHECK:     scf.yield %[[INS_DATA]], %[[INS_SCALE]]
//       CHECK:   return %[[LOOP]]#0, %[[LOOP]]#1

// -----

// `scale_dims` is empty: the `scale_dest` is a single scalar covering the whole
// tensor, so the only data dimension is reduced into it and cannot be tiled.
func.func @negative_mx_pack_no_scale_dims(%src: tensor<256xf32>,
    %data: tensor<256xi8>, %scales: tensor<f8E8M0FNU>)
    -> (tensor<256xi8>, tensor<f8E8M0FNU>) {
  // expected-error @+2 {{failed to tile operation}}
  // expected-error @+1 {{failed to generate tiling loops}}
  %0:2 = linalg.mx_pack %src
    scale_dims = []
    scale_blocks = []
    into %data, %scales
    : tensor<256xf32> -> tensor<256xi8>, tensor<f8E8M0FNU>
  return %0#0, %0#1 : tensor<256xi8>, tensor<f8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.mx_pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops = transform.structured.tile_using_for %0 tile_sizes [8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// One scale per row: `scale_dims = [0]` with a block of 1. The scaling dim 0 is
// parallel and tiled, while the reduced dim 1 stays untiled (tile size 0). The
// `scale_dest` tile offset/size collapse to the data tile values (block == 1).
func.func @mx_pack_tile_scale_per_row(%src: tensor<128x512xf32>,
    %data: tensor<128x512xi8>, %scales: tensor<128xf8E8M0FNU>)
    -> (tensor<128x512xi8>, tensor<128xf8E8M0FNU>) {
  %0:2 = linalg.mx_pack %src
    scale_dims = [0]
    scale_blocks = [1]
    into %data, %scales
    : tensor<128x512xf32> -> tensor<128x512xi8>, tensor<128xf8E8M0FNU>
  return %0#0, %0#1 : tensor<128x512xi8>, tensor<128xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.mx_pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops = transform.structured.tile_using_for %0 tile_sizes [16, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func.func @mx_pack_tile_scale_per_row(
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: tensor<128x512xf32>
//  CHECK-SAME:     %[[DATA:[a-zA-Z0-9]+]]: tensor<128x512xi8>
//  CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<128xf8E8M0FNU>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//       CHECK:   %[[LOOP:.+]]:2 = scf.for %[[IV:[a-zA-Z0-9]+]] = %[[C0]] to %[[C128]] step %[[C16]]
//  CHECK-SAME:       iter_args(%[[ITER0:.+]] = %[[DATA]], %[[ITER1:.+]] = %[[SCALES]])
//   CHECK-DAG:     %[[SRC_TILE:.+]] = tensor.extract_slice %[[SRC]][%[[IV]], 0] [16, 512] [1, 1]
//   CHECK-DAG:     %[[DATA_TILE:.+]] = tensor.extract_slice %[[ITER0]][%[[IV]], 0] [16, 512] [1, 1]
//   CHECK-DAG:     %[[SCALE_TILE:.+]] = tensor.extract_slice %[[ITER1]][%[[IV]]] [16] [1]
//       CHECK:     %[[PACK_DATA:.+]], %[[PACK_SCALE:.+]] = linalg.mx_pack %[[SRC_TILE]]
//  CHECK-SAME:         scale_dims = [0] scale_blocks = [1]
//  CHECK-SAME:         into %[[DATA_TILE]], %[[SCALE_TILE]]
//       CHECK:     %[[INS_DATA:.+]] = tensor.insert_slice %[[PACK_DATA]] into %[[ITER0]][%[[IV]], 0] [16, 512] [1, 1]
//       CHECK:     %[[INS_SCALE:.+]] = tensor.insert_slice %[[PACK_SCALE]] into %[[ITER1]][%[[IV]]] [16] [1]
//       CHECK:     scf.yield %[[INS_DATA]], %[[INS_SCALE]]
//       CHECK:   return %[[LOOP]]#0, %[[LOOP]]#1

// -----

// One scale per 64x128 block. The trailing tile of the second scaling dim is a
// partial block (500 = 3*128 + 116); the inferred `scale_dest` size is the
// `ceildiv` of the clamped data tile size, so the partial block is still
// covered by a single scale element.
func.func @mx_pack_tile_2d_block_partial(%src: tensor<128x500xf32>,
    %data: tensor<128x500xf8E5M2>, %scales: tensor<2x4xf8E8M0FNU>)
    -> (tensor<128x500xf8E5M2>, tensor<2x4xf8E8M0FNU>) {
  %0:2 = linalg.mx_pack %src
    scale_dims = [0, 1]
    scale_blocks = [64, 128]
    into %data, %scales
    : tensor<128x500xf32> -> tensor<128x500xf8E5M2>, tensor<2x4xf8E8M0FNU>
  return %0#0, %0#1 : tensor<128x500xf8E5M2>, tensor<2x4xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.mx_pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [64, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MIN:.+]] = affine_map<(d0) -> (-d0 + 500, 128)>
//   CHECK-DAG: #[[$DIV64:.+]] = affine_map<(d0) -> (d0 floordiv 64)>
//   CHECK-DAG: #[[$DIV128:.+]] = affine_map<(d0) -> (d0 floordiv 128)>
//   CHECK-DAG: #[[$CEIL128:.+]] = affine_map<(d0) -> (d0 ceildiv 128)>
// CHECK-LABEL: func.func @mx_pack_tile_2d_block_partial(
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: tensor<128x500xf32>
//  CHECK-SAME:     %[[DATA:[a-zA-Z0-9]+]]: tensor<128x500xf8E5M2>
//  CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<2x4xf8E8M0FNU>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//   CHECK-DAG:   %[[C500:.+]] = arith.constant 500 : index
//   CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//       CHECK:   %[[OUTER:.+]]:2 = scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[C128]] step %[[C64]]
//  CHECK-SAME:       iter_args(%[[ITER0_0:.+]] = %[[DATA]], %[[ITER0_1:.+]] = %[[SCALES]])
//       CHECK:     %[[INNER:.+]]:2 = scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[C500]] step %[[C128]]
//  CHECK-SAME:         iter_args(%[[ITER1_0:.+]] = %[[ITER0_0]], %[[ITER1_1:.+]] = %[[ITER0_1]])
//       CHECK:       %[[TS:.+]] = affine.min #[[$MIN]](%[[IV1]])
//   CHECK-DAG:       %[[SRC_TILE:.+]] = tensor.extract_slice %[[SRC]][%[[IV0]], %[[IV1]]] [64, %[[TS]]] [1, 1]
//   CHECK-DAG:       %[[DATA_TILE:.+]] = tensor.extract_slice %[[ITER1_0]][%[[IV0]], %[[IV1]]] [64, %[[TS]]] [1, 1]
//   CHECK-DAG:       %[[OFF0:.+]] = affine.apply #[[$DIV64]](%[[IV0]])
//   CHECK-DAG:       %[[OFF1:.+]] = affine.apply #[[$DIV128]](%[[IV1]])
//   CHECK-DAG:       %[[SZ1:.+]] = affine.apply #[[$CEIL128]](%[[TS]])
//       CHECK:       %[[SCALE_TILE:.+]] = tensor.extract_slice %[[ITER1_1]][%[[OFF0]], %[[OFF1]]] [1, %[[SZ1]]] [1, 1]
//       CHECK:       %[[PACK_DATA:.+]], %[[PACK_SCALE:.+]] = linalg.mx_pack %[[SRC_TILE]]
//  CHECK-SAME:           scale_dims = [0, 1] scale_blocks = [64, 128]
//  CHECK-SAME:           into %[[DATA_TILE]], %[[SCALE_TILE]]
//       CHECK:       %[[INS_DATA:.+]] = tensor.insert_slice %[[PACK_DATA]] into %[[ITER1_0]][%[[IV0]], %[[IV1]]] [64, %[[TS]]] [1, 1]
//       CHECK:       %[[INS_SCALE:.+]] = tensor.insert_slice %[[PACK_SCALE]] into %[[ITER1_1]][%[[OFF0]], %[[OFF1]]] [1, %[[SZ1]]] [1, 1]
//       CHECK:       scf.yield %[[INS_DATA]], %[[INS_SCALE]]
//       CHECK:     scf.yield %[[INNER]]#0, %[[INNER]]#1
//       CHECK:   return %[[OUTER]]#0, %[[OUTER]]#1

// -----

// 4-D input with two scaling dims `scale_dims = [1, 3]` and `scale_blocks =
// [4, 16]`. The scaling dims 1 and 3 are tiled to whole blocks; the reduced
// dims 0 and 2 stay untiled. The `scale_dest` tile is the 2-D block index.
func.func @mx_pack_tile_rank4(%src: tensor<2x8x3x64xf32>,
    %data: tensor<2x8x3x64xi8>, %scales: tensor<2x4xf8E8M0FNU>)
    -> (tensor<2x8x3x64xi8>, tensor<2x4xf8E8M0FNU>) {
  %0:2 = linalg.mx_pack %src
    scale_dims = [1, 3]
    scale_blocks = [4, 16]
    into %data, %scales
    : tensor<2x8x3x64xf32> -> tensor<2x8x3x64xi8>, tensor<2x4xf8E8M0FNU>
  return %0#0, %0#1 : tensor<2x8x3x64xi8>, tensor<2x4xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.mx_pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [0, 4, 0, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[$DIV4:.+]] = affine_map<(d0) -> (d0 floordiv 4)>
//   CHECK-DAG: #[[$DIV16:.+]] = affine_map<(d0) -> (d0 floordiv 16)>
// CHECK-LABEL: func.func @mx_pack_tile_rank4(
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: tensor<2x8x3x64xf32>
//  CHECK-SAME:     %[[DATA:[a-zA-Z0-9]+]]: tensor<2x8x3x64xi8>
//  CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<2x4xf8E8M0FNU>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//   CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//       CHECK:   %[[OUTER:.+]]:2 = scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[C8]] step %[[C4]]
//  CHECK-SAME:       iter_args(%[[ITER0_0:.+]] = %[[DATA]], %[[ITER0_1:.+]] = %[[SCALES]])
//       CHECK:     %[[INNER:.+]]:2 = scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[C64]] step %[[C16]]
//  CHECK-SAME:         iter_args(%[[ITER1_0:.+]] = %[[ITER0_0]], %[[ITER1_1:.+]] = %[[ITER0_1]])
//   CHECK-DAG:       %[[SRC_TILE:.+]] = tensor.extract_slice %[[SRC]][0, %[[IV0]], 0, %[[IV1]]] [2, 4, 3, 16] [1, 1, 1, 1]
//   CHECK-DAG:       %[[DATA_TILE:.+]] = tensor.extract_slice %[[ITER1_0]][0, %[[IV0]], 0, %[[IV1]]] [2, 4, 3, 16] [1, 1, 1, 1]
//   CHECK-DAG:       %[[OFF0:.+]] = affine.apply #[[$DIV4]](%[[IV0]])
//   CHECK-DAG:       %[[OFF1:.+]] = affine.apply #[[$DIV16]](%[[IV1]])
//       CHECK:       %[[SCALE_TILE:.+]] = tensor.extract_slice %[[ITER1_1]][%[[OFF0]], %[[OFF1]]] [1, 1] [1, 1]
//       CHECK:       %[[PACK_DATA:.+]], %[[PACK_SCALE:.+]] = linalg.mx_pack %[[SRC_TILE]]
//  CHECK-SAME:           scale_dims = [1, 3] scale_blocks = [4, 16]
//  CHECK-SAME:           into %[[DATA_TILE]], %[[SCALE_TILE]]
//       CHECK:       %[[INS_DATA:.+]] = tensor.insert_slice %[[PACK_DATA]] into %[[ITER1_0]][0, %[[IV0]], 0, %[[IV1]]] [2, 4, 3, 16] [1, 1, 1, 1]
//       CHECK:       %[[INS_SCALE:.+]] = tensor.insert_slice %[[PACK_SCALE]] into %[[ITER1_1]][%[[OFF0]], %[[OFF1]]] [1, 1] [1, 1]
//       CHECK:       scf.yield %[[INS_DATA]], %[[INS_SCALE]]
//       CHECK:     scf.yield %[[INNER]]#0, %[[INNER]]#1
//       CHECK:   return %[[OUTER]]#0, %[[OUTER]]#1

// -----

// Same 4-D op, but the tile size 8 on scaling dim 3 is neither equal to the
// dimension (64) nor a multiple of the scale block (16). Splitting a scale
// block across tiles is not supported, so the op is not tiled.
func.func @negative_mx_pack_partial_scale_block_tiling(%src: tensor<2x8x3x64xf32>,
    %data: tensor<2x8x3x64xi8>, %scales: tensor<2x4xf8E8M0FNU>)
    -> (tensor<2x8x3x64xi8>, tensor<2x4xf8E8M0FNU>) {
  // expected-error @+2 {{failed to tile operation}}
  // expected-error @+1 {{failed to generate tiling loops}}
  %0:2 = linalg.mx_pack %src
    scale_dims = [1, 3]
    scale_blocks = [4, 16]
    into %data, %scales
    : tensor<2x8x3x64xf32> -> tensor<2x8x3x64xi8>, tensor<2x4xf8E8M0FNU>
  return %0#0, %0#1 : tensor<2x8x3x64xi8>, tensor<2x4xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.mx_pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [0, 4, 0, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
