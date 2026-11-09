// RUN: mlir-opt %s -transform-preload-library='transform-library-paths=%p/td/unroll-scaled-contract.mlir' \
// RUN: -transform-interpreter=entry-point=unroll_scaled_contract --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Test UnrollScaledContractionPattern
//===----------------------------------------------------------------------===//

// Tiles along the block scaled dim span multiple blocks. Tiles along the
// reduction dim accumulate into the same accumulator slice.

// CHECK-LABEL: @scaled_matmul_tile_multiple_of_block
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: vector<4x32xf8E4M3FN>, %[[LHS_SCALE:[a-zA-Z0-9]+]]: vector<4x4xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: vector<8x32xf8E4M3FN>, %[[RHS_SCALE:[a-zA-Z0-9]+]]: vector<8x4xf8E8M0FNU>
//  CHECK-SAME:   %[[ACC:[a-zA-Z0-9]+]]: vector<4x8xf32>
//       CHECK:   %[[LHS_0:.+]] = vector.extract_strided_slice %[[LHS]] offsets = [0, 0], sizes = [2, 16], strides = [1, 1]
//       CHECK:   %[[LHS_SCALE_0:.+]] = vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0, 0], sizes = [2, 2], strides = [1, 1]
//       CHECK:   %[[RHS_0:.+]] = vector.extract_strided_slice %[[RHS]] offsets = [0, 0], sizes = [4, 16], strides = [1, 1]
//       CHECK:   %[[RHS_SCALE_0:.+]] = vector.extract_strided_slice %[[RHS_SCALE]] offsets = [0, 0], sizes = [4, 2], strides = [1, 1]
//       CHECK:   %[[ACC_0:.+]] = vector.extract_strided_slice %[[ACC]] offsets = [0, 0], sizes = [2, 4], strides = [1, 1]
//       CHECK:   %[[RES_0:.+]] = vector.scaled_contract {{.*}} %[[LHS_0]], %[[LHS_SCALE_0]], %[[RHS_0]], %[[RHS_SCALE_0]], %[[ACC_0]]
//  CHECK-SAME:     : vector<2x16xf8E4M3FN>, vector<2x2xf8E8M0FNU>, vector<4x16xf8E4M3FN>, vector<4x2xf8E8M0FNU> into vector<2x4xf32>
//       CHECK:   %[[LHS_1:.+]] = vector.extract_strided_slice %[[LHS]] offsets = [0, 16], sizes = [2, 16], strides = [1, 1]
//       CHECK:   %[[LHS_SCALE_1:.+]] = vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0, 2], sizes = [2, 2], strides = [1, 1]
//       CHECK:   %[[RHS_1:.+]] = vector.extract_strided_slice %[[RHS]] offsets = [0, 16], sizes = [4, 16], strides = [1, 1]
//       CHECK:   %[[RHS_SCALE_1:.+]] = vector.extract_strided_slice %[[RHS_SCALE]] offsets = [0, 2], sizes = [4, 2], strides = [1, 1]
//       CHECK:   vector.scaled_contract {{.*}} %[[LHS_1]], %[[LHS_SCALE_1]], %[[RHS_1]], %[[RHS_SCALE_1]], %[[RES_0]]
//  CHECK-COUNT-6:   vector.scaled_contract {{.*}} : vector<2x16xf8E4M3FN>, vector<2x2xf8E8M0FNU>, vector<4x16xf8E4M3FN>, vector<4x2xf8E8M0FNU> into vector<2x4xf32>
func.func @scaled_matmul_tile_multiple_of_block(
    %lhs: vector<4x32xf8E4M3FN>, %lhs_scale: vector<4x4xf8E8M0FNU>,
    %rhs: vector<8x32xf8E4M3FN>, %rhs_scale: vector<8x4xf8E8M0FNU>,
    %acc: vector<4x8xf32>) -> vector<4x8xf32> {
  %res = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 8)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 8)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %lhs, %lhs_scale, %rhs, %rhs_scale, %acc
    : vector<4x32xf8E4M3FN>, vector<4x4xf8E8M0FNU>,
      vector<8x32xf8E4M3FN>, vector<8x4xf8E8M0FNU> into vector<4x8xf32>
  return %res : vector<4x8xf32>
}

// -----

// Tiles along the block scaled dim span a single block each.

// CHECK-LABEL: @scaled_matmul_tile_equal_to_block
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: vector<4x32xf4E2M1FN>, %[[LHS_SCALE:[a-zA-Z0-9]+]]: vector<4x2xf8E4M3FN>
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0, 0], sizes = [2, 1], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0, 1], sizes = [2, 1], strides = [1, 1]
//       CHECK-COUNT-7:   vector.scaled_contract {{.*}} : vector<2x16xf4E2M1FN>, vector<2x1xf8E4M3FN>, vector<4x16xf4E2M1FN>, vector<4x1xf8E4M3FN> into vector<2x4xf32>
func.func @scaled_matmul_tile_equal_to_block(
    %lhs: vector<4x32xf4E2M1FN>, %lhs_scale: vector<4x2xf8E4M3FN>,
    %rhs: vector<8x32xf4E2M1FN>, %rhs_scale: vector<8x2xf8E4M3FN>,
    %acc: vector<4x8xf32>) -> vector<4x8xf32> {
  %res = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 16)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 16)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %lhs, %lhs_scale, %rhs, %rhs_scale, %acc
    : vector<4x32xf4E2M1FN>, vector<4x2xf8E4M3FN>,
      vector<8x32xf4E2M1FN>, vector<8x2xf8E4M3FN> into vector<4x8xf32>
  return %res : vector<4x8xf32>
}

// -----

// Tiles along the block scaled dim fit within blocks and share their scales.

// CHECK-LABEL: @scaled_matmul_tile_within_block
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: vector<2x64xf8E5M2>, %[[LHS_SCALE:[a-zA-Z0-9]+]]: vector<2x2xf8E8M0FNU>
//       CHECK:   vector.extract_strided_slice %[[LHS]] offsets = [0, 0], sizes = [2, 16], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0, 0], sizes = [2, 1], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS]] offsets = [0, 16], sizes = [2, 16], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0, 0], sizes = [2, 1], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS]] offsets = [0, 32], sizes = [2, 16], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0, 1], sizes = [2, 1], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS]] offsets = [0, 48], sizes = [2, 16], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0, 1], sizes = [2, 1], strides = [1, 1]
//       CHECK:   %[[RES:.+]] = vector.scaled_contract
//  CHECK-SAME:     : vector<2x16xf8E5M2>, vector<2x1xf8E8M0FNU>, vector<4x16xf8E5M2>, vector<4x1xf8E8M0FNU> into vector<2x4xf32>
//       CHECK:   return %[[RES]]
func.func @scaled_matmul_tile_within_block(
    %lhs: vector<2x64xf8E5M2>, %lhs_scale: vector<2x2xf8E8M0FNU>,
    %rhs: vector<4x64xf8E5M2>, %rhs_scale: vector<4x2xf8E8M0FNU>,
    %acc: vector<2x4xf32>) -> vector<2x4xf32> {
  %res = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 32)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 32)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %lhs, %lhs_scale, %rhs, %rhs_scale, %acc
    : vector<2x64xf8E5M2>, vector<2x2xf8E8M0FNU>,
      vector<4x64xf8E5M2>, vector<4x2xf8E8M0FNU> into vector<2x4xf32>
  return %res : vector<2x4xf32>
}

// -----

// A single block spanning the whole dim is reused by every tile.

// CHECK-LABEL: @scaled_matmul_single_block
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: vector<2x48xf8E4M3FN>, %[[LHS_SCALE:[a-zA-Z0-9]+]]: vector<2x1xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: vector<4x48xf8E4M3FN>, %[[RHS_SCALE:[a-zA-Z0-9]+]]: vector<4x1xf8E8M0FNU>
//  CHECK-COUNT-3:   vector.scaled_contract {{.*}}, %[[LHS_SCALE]], %{{.+}}, %[[RHS_SCALE]], %{{.+}} : vector<2x16xf8E4M3FN>, vector<2x1xf8E8M0FNU>, vector<4x16xf8E4M3FN>, vector<4x1xf8E8M0FNU> into vector<2x4xf32>
func.func @scaled_matmul_single_block(
    %lhs: vector<2x48xf8E4M3FN>, %lhs_scale: vector<2x1xf8E8M0FNU>,
    %rhs: vector<4x48xf8E4M3FN>, %rhs_scale: vector<4x1xf8E8M0FNU>,
    %acc: vector<2x4xf32>) -> vector<2x4xf32> {
  %res = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 64)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 64)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %lhs, %lhs_scale, %rhs, %rhs_scale, %acc
    : vector<2x48xf8E4M3FN>, vector<2x1xf8E8M0FNU>,
      vector<4x48xf8E4M3FN>, vector<4x1xf8E8M0FNU> into vector<2x4xf32>
  return %res : vector<2x4xf32>
}

// -----

// Dimension scales are sliced together with their parallel dims.

// CHECK-LABEL: @scaled_matmul_dim_scaling
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: vector<4x16xi8>, %[[LHS_SCALE:[a-zA-Z0-9]+]]: vector<4xf32>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: vector<8x16xi8>, %[[RHS_SCALE:[a-zA-Z0-9]+]]: vector<8xf32>
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0], sizes = [2], strides = [1]
//       CHECK:   vector.extract_strided_slice %[[RHS_SCALE]] offsets = [0], sizes = [4], strides = [1]
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0], sizes = [2], strides = [1]
//       CHECK:   vector.extract_strided_slice %[[RHS_SCALE]] offsets = [4], sizes = [4], strides = [1]
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [2], sizes = [2], strides = [1]
//       CHECK:   vector.extract_strided_slice %[[RHS_SCALE]] offsets = [0], sizes = [4], strides = [1]
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [2], sizes = [2], strides = [1]
//       CHECK:   vector.extract_strided_slice %[[RHS_SCALE]] offsets = [4], sizes = [4], strides = [1]
//       CHECK:   vector.scaled_contract {{.*}} : vector<2x16xi8>, vector<2xf32>, vector<4x16xi8>, vector<4xf32> into vector<2x4xf32>
func.func @scaled_matmul_dim_scaling(
    %lhs: vector<4x16xi8>, %lhs_scale: vector<4xf32>,
    %rhs: vector<8x16xi8>, %rhs_scale: vector<8xf32>,
    %acc: vector<4x8xf32>) -> vector<4x8xf32> {
  %res = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %lhs, %lhs_scale, %rhs, %rhs_scale, %acc
    : vector<4x16xi8>, vector<4xf32>,
      vector<8x16xi8>, vector<8xf32> into vector<4x8xf32>
  return %res : vector<4x8xf32>
}

// -----

// Transposed block scale is sliced along its block dim and a scalar scale is
// reused by every tile.

// CHECK-LABEL: @scaled_matmul_transposed_scale
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: vector<2x32xf8E4M3FN>, %[[LHS_SCALE:[a-zA-Z0-9]+]]: f32
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: vector<32x4xf8E4M3FN>, %[[RHS_SCALE:[a-zA-Z0-9]+]]: vector<2x4xf8E8M0FNU>
//       CHECK:   %[[RHS_SCALE_0:.+]] = vector.extract_strided_slice %[[RHS_SCALE]] offsets = [0, 0], sizes = [1, 4], strides = [1, 1]
//       CHECK:   vector.scaled_contract {{.*}} %{{.+}}, %[[LHS_SCALE]], %{{.+}}, %[[RHS_SCALE_0]], %{{.+}} : vector<2x16xf8E4M3FN>, f32, vector<16x4xf8E4M3FN>, vector<1x4xf8E8M0FNU> into vector<2x4xf32>
//       CHECK:   %[[RHS_SCALE_1:.+]] = vector.extract_strided_slice %[[RHS_SCALE]] offsets = [1, 0], sizes = [1, 4], strides = [1, 1]
//       CHECK:   vector.scaled_contract {{.*}} %{{.+}}, %[[LHS_SCALE]], %{{.+}}, %[[RHS_SCALE_1]], %{{.+}} : vector<2x16xf8E4M3FN>, f32, vector<16x4xf8E4M3FN>, vector<1x4xf8E8M0FNU> into vector<2x4xf32>
func.func @scaled_matmul_transposed_scale(
    %lhs: vector<2x32xf8E4M3FN>, %lhs_scale: f32,
    %rhs: vector<32x4xf8E4M3FN>, %rhs_scale: vector<2x4xf8E8M0FNU>,
    %acc: vector<2x4xf32>) -> vector<2x4xf32> {
  %res = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> ()>,
                     affine_map<(m, n, k) -> (k, n)>,
                     affine_map<(m, n, k) -> (k floordiv 16, n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %lhs, %lhs_scale, %rhs, %rhs_scale, %acc
    : vector<2x32xf8E4M3FN>, f32,
      vector<32x4xf8E4M3FN>, vector<2x4xf8E8M0FNU> into vector<2x4xf32>
  return %res : vector<2x4xf32>
}

// -----

// Block scaling of a parallel dim - scale blocks are sliced with the tiles.

// CHECK-LABEL: @scaled_matmul_parallel_block
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: vector<4x16xf8E4M3FN>, %[[LHS_SCALE:[a-zA-Z0-9]+]]: vector<2x1xf8E8M0FNU>
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0, 0], sizes = [1, 1], strides = [1, 1]
//       CHECK:   vector.scaled_contract {{.*}} : vector<2x16xf8E4M3FN>, vector<1x1xf8E8M0FNU>, vector<4x16xf8E4M3FN>, f32 into vector<2x4xf32>
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [1, 0], sizes = [1, 1], strides = [1, 1]
//       CHECK:   vector.scaled_contract {{.*}} : vector<2x16xf8E4M3FN>, vector<1x1xf8E8M0FNU>, vector<4x16xf8E4M3FN>, f32 into vector<2x4xf32>
func.func @scaled_matmul_parallel_block(
    %lhs: vector<4x16xf8E4M3FN>, %lhs_scale: vector<2x1xf8E8M0FNU>,
    %rhs: vector<4x16xf8E4M3FN>, %rhs_scale: f32,
    %acc: vector<4x4xf32>) -> vector<4x4xf32> {
  %res = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m floordiv 2, k floordiv 16)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> ()>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %lhs, %lhs_scale, %rhs, %rhs_scale, %acc
    : vector<4x16xf8E4M3FN>, vector<2x1xf8E8M0FNU>,
      vector<4x16xf8E4M3FN>, f32 into vector<4x4xf32>
  return %res : vector<4x4xf32>
}

// -----

// Different block sizes of the inputs - tiles fit within blocks of both.

// CHECK-LABEL: @scaled_matmul_different_blocks
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: vector<2x96xf8E4M3FN>, %[[LHS_SCALE:[a-zA-Z0-9]+]]: vector<2x3xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: vector<4x96xf8E4M3FN>, %[[RHS_SCALE:[a-zA-Z0-9]+]]: vector<4x2xf8E8M0FNU>
//       CHECK:   vector.extract_strided_slice %[[LHS]] offsets = [0, 32], sizes = [2, 16], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0, 1], sizes = [2, 1], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS]] offsets = [0, 32], sizes = [4, 16], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS_SCALE]] offsets = [0, 0], sizes = [4, 1], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS]] offsets = [0, 48], sizes = [2, 16], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS_SCALE]] offsets = [0, 1], sizes = [2, 1], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS]] offsets = [0, 48], sizes = [4, 16], strides = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS_SCALE]] offsets = [0, 1], sizes = [4, 1], strides = [1, 1]
func.func @scaled_matmul_different_blocks(
    %lhs: vector<2x96xf8E4M3FN>, %lhs_scale: vector<2x3xf8E8M0FNU>,
    %rhs: vector<4x96xf8E4M3FN>, %rhs_scale: vector<4x2xf8E8M0FNU>,
    %acc: vector<2x4xf32>) -> vector<2x4xf32> {
  %res = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 32)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 48)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %lhs, %lhs_scale, %rhs, %rhs_scale, %acc
    : vector<2x96xf8E4M3FN>, vector<2x3xf8E8M0FNU>,
      vector<4x96xf8E4M3FN>, vector<4x2xf8E8M0FNU> into vector<2x4xf32>
  return %res : vector<2x4xf32>
}

// -----

// Tiles would split scale blocks.

// CHECK-LABEL: @negative_scaled_matmul_tile_splits_blocks
//   CHECK-NOT:   vector.extract_strided_slice
//       CHECK:   vector.scaled_contract {{.*}} : vector<2x48xf8E4M3FN>, vector<2x2xf8E8M0FNU>, vector<4x48xf8E4M3FN>, vector<4x2xf8E8M0FNU> into vector<2x4xf32>
func.func @negative_scaled_matmul_tile_splits_blocks(
    %lhs: vector<2x48xf8E4M3FN>, %lhs_scale: vector<2x2xf8E8M0FNU>,
    %rhs: vector<4x48xf8E4M3FN>, %rhs_scale: vector<4x2xf8E8M0FNU>,
    %acc: vector<2x4xf32>) -> vector<2x4xf32> {
  %res = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 24)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 24)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %lhs, %lhs_scale, %rhs, %rhs_scale, %acc
    : vector<2x48xf8E4M3FN>, vector<2x2xf8E8M0FNU>,
      vector<4x48xf8E4M3FN>, vector<4x2xf8E8M0FNU> into vector<2x4xf32>
  return %res : vector<2x4xf32>
}

// -----

// The transform op only unrolls matmul-like ops.

// CHECK-LABEL: @negative_scaled_batched_matmul
//   CHECK-NOT:   vector.extract_strided_slice
//       CHECK:   vector.scaled_contract {{.*}} : vector<1x2x32xf8E4M3FN>, vector<1x2x1xf8E8M0FNU>, vector<1x4x32xf8E4M3FN>, vector<1x4x1xf8E8M0FNU> into vector<1x2x4xf32>
func.func @negative_scaled_batched_matmul(
    %lhs: vector<1x2x32xf8E4M3FN>, %lhs_scale: vector<1x2x1xf8E8M0FNU>,
    %rhs: vector<1x4x32xf8E4M3FN>, %rhs_scale: vector<1x4x1xf8E8M0FNU>,
    %acc: vector<1x2x4xf32>) -> vector<1x2x4xf32> {
  %res = vector.scaled_contract {
    indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                     affine_map<(b, m, n, k) -> (b, m, k floordiv 32)>,
                     affine_map<(b, m, n, k) -> (b, n, k)>,
                     affine_map<(b, m, n, k) -> (b, n, k floordiv 32)>,
                     affine_map<(b, m, n, k) -> (b, m, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    %lhs, %lhs_scale, %rhs, %rhs_scale, %acc
    : vector<1x2x32xf8E4M3FN>, vector<1x2x1xf8E8M0FNU>,
      vector<1x4x32xf8E4M3FN>, vector<1x4x1xf8E8M0FNU> into vector<1x2x4xf32>
  return %res : vector<1x2x4xf32>
}
