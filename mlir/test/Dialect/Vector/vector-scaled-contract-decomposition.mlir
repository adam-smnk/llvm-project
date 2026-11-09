// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[$MAP_ACC_T:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK-DAG: #[[$MAP_RHS_T:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP_BLK_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$MAP_BLK_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[$MAP_BLK_PARTIAL:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP_NESTED_LHS:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>
// CHECK-DAG: #[[$MAP_NESTED_RHS:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[$MAP_NESTED_PARTIAL:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[$MAP_DOT_IN:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP_DOT_PARTIAL:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[$MAP_BATCH_LHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[$MAP_BATCH_RHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4, d2)>
// CHECK-DAG: #[[$MAP_BATCH_PARTIAL:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP_PRESCALE_BATCH_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>

//===----------------------------------------------------------------------===//
// [Pattern: DecomposeScaledContractByBlocks]
//===----------------------------------------------------------------------===//

// Dimension scaling of parallel dims - narrow integers are contracted into i32
// and the scales are applied to the contraction results.

// CHECK-LABEL: func @scaled_contract_dim_scaling
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<16x64xi8>, %[[SA:[a-zA-Z0-9]+]]: vector<16xf32>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<8x64xi8>, %[[SB:[a-zA-Z0-9]+]]: vector<8xf32>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<16x8xf32>
//       CHECK:   %[[ZERO:.+]] = arith.constant dense<0> : vector<16x8xi32>
//       CHECK:   %[[CONTRACT:.+]] = vector.contract
//  CHECK-SAME:     indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     %[[A]], %[[B]], %[[ZERO]] : vector<16x64xi8>, vector<8x64xi8> into vector<16x8xi32>
//       CHECK:   %[[PARTIAL:.+]] = arith.sitofp %[[CONTRACT]] : vector<16x8xi32> to vector<16x8xf32>
//       CHECK:   %[[SA_CAST:.+]] = vector.shape_cast %[[SA]] : vector<16xf32> to vector<16x1xf32>
//       CHECK:   %[[SA_BCAST:.+]] = vector.broadcast %[[SA_CAST]] : vector<16x1xf32> to vector<16x8xf32>
//       CHECK:   %[[MUL_SA:.+]] = arith.mulf %[[PARTIAL]], %[[SA_BCAST]] : vector<16x8xf32>
//       CHECK:   %[[SB_BCAST:.+]] = vector.broadcast %[[SB]] : vector<8xf32> to vector<16x8xf32>
//       CHECK:   %[[MUL_SB:.+]] = arith.mulf %[[MUL_SA]], %[[SB_BCAST]] : vector<16x8xf32>
//       CHECK:   %[[RES:.+]] = arith.addf %[[C]], %[[MUL_SB]] : vector<16x8xf32>
//       CHECK:   return %[[RES]] : vector<16x8xf32>
func.func @scaled_contract_dim_scaling(
    %A: vector<16x64xi8>, %sA: vector<16xf32>,
    %B: vector<8x64xi8>, %sB: vector<8xf32>,
    %C: vector<16x8xf32>) -> vector<16x8xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<16x64xi8>, vector<16xf32>,
      vector<8x64xi8>, vector<8xf32> into vector<16x8xf32>
  return %0 : vector<16x8xf32>
}

// MXFP4 block scaling of the reduction dim - the reduction dim is split into
// blocks, the inputs are contracted in low precision within the blocks, and
// the scaled block results are reduced into the accumulator.

// CHECK-LABEL: func @scaled_contract_mxfp4
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<8x64xf4E2M1FN>, %[[SA:[a-zA-Z0-9]+]]: vector<8x2xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<4x64xf4E2M1FN>, %[[SB:[a-zA-Z0-9]+]]: vector<4x2xf8E8M0FNU>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<8x4xf32>
//       CHECK:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<8x4x2xf32>
//       CHECK:   %[[A_BLK:.+]] = vector.shape_cast %[[A]] : vector<8x64xf4E2M1FN> to vector<8x2x32xf4E2M1FN>
//       CHECK:   %[[B_BLK:.+]] = vector.shape_cast %[[B]] : vector<4x64xf4E2M1FN> to vector<4x2x32xf4E2M1FN>
//       CHECK:   %[[PARTIAL:.+]] = vector.contract
//  CHECK-SAME:     indexing_maps = [#[[$MAP_BLK_LHS]], #[[$MAP_BLK_RHS]], #[[$MAP_BLK_PARTIAL]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:     %[[A_BLK]], %[[B_BLK]], %[[ZERO]] : vector<8x2x32xf4E2M1FN>, vector<4x2x32xf4E2M1FN> into vector<8x4x2xf32>
//       CHECK:   %[[SA_EXT:.+]] = arith.extf %[[SA]] : vector<8x2xf8E8M0FNU> to vector<8x2xf32>
//       CHECK:   %[[SA_CAST:.+]] = vector.shape_cast %[[SA_EXT]] : vector<8x2xf32> to vector<8x1x2xf32>
//       CHECK:   %[[SA_BCAST:.+]] = vector.broadcast %[[SA_CAST]] : vector<8x1x2xf32> to vector<8x4x2xf32>
//       CHECK:   %[[MUL_SA:.+]] = arith.mulf %[[PARTIAL]], %[[SA_BCAST]] : vector<8x4x2xf32>
//       CHECK:   %[[SB_EXT:.+]] = arith.extf %[[SB]] : vector<4x2xf8E8M0FNU> to vector<4x2xf32>
//       CHECK:   %[[SB_BCAST:.+]] = vector.broadcast %[[SB_EXT]] : vector<4x2xf32> to vector<8x4x2xf32>
//       CHECK:   %[[MUL_SB:.+]] = arith.mulf %[[MUL_SA]], %[[SB_BCAST]] : vector<8x4x2xf32>
//       CHECK:   %[[RES:.+]] = vector.multi_reduction <add>, %[[MUL_SB]], %[[C]] [2] : vector<8x4x2xf32> to vector<8x4xf32>
//       CHECK:   return %[[RES]] : vector<8x4xf32>
func.func @scaled_contract_mxfp4(
    %A: vector<8x64xf4E2M1FN>, %sA: vector<8x2xf8E8M0FNU>,
    %B: vector<4x64xf4E2M1FN>, %sB: vector<4x2xf8E8M0FNU>,
    %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 32)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 32)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<8x64xf4E2M1FN>, vector<8x2xf8E8M0FNU>,
      vector<4x64xf4E2M1FN>, vector<4x2xf8E8M0FNU> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// Different narrow floating-point inputs are extended as vector.contract
// requires inputs of the same type.

// CHECK-LABEL: func @scaled_contract_mixed_narrow_floats
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<8x64xf8E5M2>, %[[SA:[a-zA-Z0-9]+]]: vector<8x2xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<4x64xf8E4M3FN>
//       CHECK:   %[[A_BLK:.+]] = vector.shape_cast %[[A]] : vector<8x64xf8E5M2> to vector<8x2x32xf8E5M2>
//       CHECK:   %[[B_BLK:.+]] = vector.shape_cast %[[B]] : vector<4x64xf8E4M3FN> to vector<4x2x32xf8E4M3FN>
//       CHECK:   %[[A_EXT:.+]] = arith.extf %[[A_BLK]] : vector<8x2x32xf8E5M2> to vector<8x2x32xf32>
//       CHECK:   %[[B_EXT:.+]] = arith.extf %[[B_BLK]] : vector<4x2x32xf8E4M3FN> to vector<4x2x32xf32>
//       CHECK:   vector.contract
//  CHECK-SAME:     %[[A_EXT]], %[[B_EXT]], %{{.+}} : vector<8x2x32xf32>, vector<4x2x32xf32> into vector<8x4x2xf32>
//       CHECK:   vector.multi_reduction <add>
func.func @scaled_contract_mixed_narrow_floats(
    %A: vector<8x64xf8E5M2>, %sA: vector<8x2xf8E8M0FNU>,
    %B: vector<4x64xf8E4M3FN>, %sB: vector<4x2xf8E8M0FNU>,
    %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 32)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 32)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<8x64xf8E5M2>, vector<8x2xf8E8M0FNU>,
      vector<4x64xf8E4M3FN>, vector<4x2xf8E8M0FNU> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// NVFP4 block scaling - scales other than f8E8M0FNU are applied exactly.

// CHECK-LABEL: func @scaled_contract_nvfp4
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<8x32xf4E2M1FN>, %[[SA:[a-zA-Z0-9]+]]: vector<8x2xf8E4M3FN>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<4x32xf4E2M1FN>, %[[SB:[a-zA-Z0-9]+]]: vector<4x2xf8E4M3FN>
//       CHECK:   vector.contract
//  CHECK-SAME:     : vector<8x2x16xf4E2M1FN>, vector<4x2x16xf4E2M1FN> into vector<8x4x2xf32>
//   CHECK-NOT:   arith.truncf
//       CHECK:   arith.extf %[[SA]] : vector<8x2xf8E4M3FN> to vector<8x2xf32>
//   CHECK-NOT:   arith.truncf
//       CHECK:   arith.extf %[[SB]] : vector<4x2xf8E4M3FN> to vector<4x2xf32>
//   CHECK-NOT:   arith.truncf
//       CHECK:   vector.multi_reduction <add>
func.func @scaled_contract_nvfp4(
    %A: vector<8x32xf4E2M1FN>, %sA: vector<8x2xf8E4M3FN>,
    %B: vector<4x32xf4E2M1FN>, %sB: vector<4x2xf8E4M3FN>,
    %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 16)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 16)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<8x32xf4E2M1FN>, vector<8x2xf8E4M3FN>,
      vector<4x32xf4E2M1FN>, vector<4x2xf8E4M3FN> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// Nested block sizes of the reduction dim and block scaling of a parallel dim.
// The dims are split into multiple block levels, and the scales and the
// accumulator are reshaped accordingly. Output narrower than f32 is accumulated
// in f32 and rounded once.

// CHECK-LABEL: func @scaled_contract_nested_blocks
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<8x128xi8>, %[[SA:[a-zA-Z0-9]+]]: vector<8x4xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<4x128xi8>, %[[SB:[a-zA-Z0-9]+]]: vector<2x2xf16>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<8x4xf16>
//   CHECK-DAG:   %[[ZERO_SUM:.+]] = arith.constant dense<0.000000e+00> : vector<8x2x2xf32>
//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0> : vector<8x2x2x2x2xi32>
//       CHECK:   %[[A_BLK:.+]] = vector.shape_cast %[[A]] : vector<8x128xi8> to vector<8x2x2x32xi8>
//       CHECK:   %[[B_BLK:.+]] = vector.shape_cast %[[B]] : vector<4x128xi8> to vector<2x2x2x2x32xi8>
//       CHECK:   %[[CONTRACT:.+]] = vector.contract
//  CHECK-SAME:     indexing_maps = [#[[$MAP_NESTED_LHS]], #[[$MAP_NESTED_RHS]], #[[$MAP_NESTED_PARTIAL]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:     %[[A_BLK]], %[[B_BLK]], %[[ZERO]] : vector<8x2x2x32xi8>, vector<2x2x2x2x32xi8> into vector<8x2x2x2x2xi32>
//       CHECK:   %[[PARTIAL:.+]] = arith.sitofp %[[CONTRACT]] : vector<8x2x2x2x2xi32> to vector<8x2x2x2x2xf32>
//       CHECK:   %[[SA_BLK:.+]] = vector.shape_cast %[[SA]] : vector<8x4xf8E8M0FNU> to vector<8x2x2xf8E8M0FNU>
//       CHECK:   %[[SA_EXT:.+]] = arith.extf %[[SA_BLK]] : vector<8x2x2xf8E8M0FNU> to vector<8x2x2xf32>
//       CHECK:   %[[SA_CAST:.+]] = vector.shape_cast %[[SA_EXT]] : vector<8x2x2xf32> to vector<8x1x1x2x2xf32>
//       CHECK:   %[[SA_BCAST:.+]] = vector.broadcast %[[SA_CAST]] : vector<8x1x1x2x2xf32> to vector<8x2x2x2x2xf32>
//       CHECK:   %[[MUL_SA:.+]] = arith.mulf %[[PARTIAL]], %[[SA_BCAST]] : vector<8x2x2x2x2xf32>
//       CHECK:   %[[SB_EXT:.+]] = arith.extf %[[SB]] : vector<2x2xf16> to vector<2x2xf32>
//       CHECK:   %[[SB_CAST:.+]] = vector.shape_cast %[[SB_EXT]] : vector<2x2xf32> to vector<1x2x1x2x1xf32>
//       CHECK:   %[[SB_BCAST:.+]] = vector.broadcast %[[SB_CAST]] : vector<1x2x1x2x1xf32> to vector<8x2x2x2x2xf32>
//       CHECK:   %[[MUL_SB:.+]] = arith.mulf %[[MUL_SA]], %[[SB_BCAST]] : vector<8x2x2x2x2xf32>
//       CHECK:   %[[C_BLK:.+]] = vector.shape_cast %[[C]] : vector<8x4xf16> to vector<8x2x2xf16>
//       CHECK:   %[[SUM:.+]] = vector.multi_reduction <add>, %[[MUL_SB]], %[[ZERO_SUM]] [3, 4] : vector<8x2x2x2x2xf32> to vector<8x2x2xf32>
//       CHECK:   %[[C_EXT:.+]] = arith.extf %[[C_BLK]] : vector<8x2x2xf16> to vector<8x2x2xf32>
//       CHECK:   %[[ACC:.+]] = arith.addf %[[C_EXT]], %[[SUM]] : vector<8x2x2xf32>
//       CHECK:   %[[TRUNC:.+]] = arith.truncf %[[ACC]] : vector<8x2x2xf32> to vector<8x2x2xf16>
//       CHECK:   %[[RES:.+]] = vector.shape_cast %[[TRUNC]] : vector<8x2x2xf16> to vector<8x4xf16>
//       CHECK:   return %[[RES]] : vector<8x4xf16>
func.func @scaled_contract_nested_blocks(
    %A: vector<8x128xi8>, %sA: vector<8x4xf8E8M0FNU>,
    %B: vector<4x128xi8>, %sB: vector<2x2xf16>,
    %C: vector<8x4xf16>) -> vector<8x4xf16> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 32)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n floordiv 2, k floordiv 64)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<8x128xi8>, vector<8x4xf8E8M0FNU>,
      vector<4x128xi8>, vector<2x2xf16> into vector<8x4xf16>
  return %0 : vector<8x4xf16>
}

// A single block spanning the whole reduction dim, e.g., after unrolling, has
// a single scale which is applied after the whole contraction.

// CHECK-LABEL: func @scaled_contract_single_block
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<8x24xf8E4M3FN>, %[[SA:[a-zA-Z0-9]+]]: vector<8x1xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<4x24xf8E4M3FN>, %[[SB:[a-zA-Z0-9]+]]: vector<4x1xf8E8M0FNU>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<8x4xf32>
//       CHECK:   vector.contract
//  CHECK-SAME:     indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
//  CHECK-SAME:     %[[A]], %[[B]], %{{.+}} : vector<8x24xf8E4M3FN>, vector<4x24xf8E4M3FN> into vector<8x4xf32>
//       CHECK:   %[[SA_CAST:.+]] = vector.shape_cast %[[SA]] : vector<8x1xf8E8M0FNU> to vector<8xf8E8M0FNU>
//       CHECK:   arith.extf %[[SA_CAST]] : vector<8xf8E8M0FNU> to vector<8xf32>
//       CHECK:   %[[SB_CAST:.+]] = vector.shape_cast %[[SB]] : vector<4x1xf8E8M0FNU> to vector<4xf8E8M0FNU>
//       CHECK:   arith.extf %[[SB_CAST]] : vector<4xf8E8M0FNU> to vector<4xf32>
//       CHECK:   %[[SCALED:.+]] = arith.mulf {{.+}} : vector<8x4xf32>
//       CHECK:   %[[RES:.+]] = arith.addf %[[C]], %[[SCALED]] : vector<8x4xf32>
//       CHECK:   return %[[RES]]
func.func @scaled_contract_single_block(
    %A: vector<8x24xf8E4M3FN>, %sA: vector<8x1xf8E8M0FNU>,
    %B: vector<4x24xf8E4M3FN>, %sB: vector<4x1xf8E8M0FNU>,
    %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 32)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 32)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<8x24xf8E4M3FN>, vector<8x1xf8E8M0FNU>,
      vector<4x24xf8E4M3FN>, vector<4x1xf8E8M0FNU> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// Tensor scaling with scalar scales and transposed output.

// CHECK-LABEL: func @scaled_contract_tensor_scaling
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<8x64xbf16>, %[[SA:[a-zA-Z0-9]+]]: f32
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<4x64xbf16>, %[[SB:[a-zA-Z0-9]+]]: f32
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<4x8xf32>
//       CHECK:   %[[PARTIAL:.+]] = vector.contract
//  CHECK-SAME:     indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC_T]]]
//  CHECK-SAME:     %[[A]], %[[B]], %{{.+}} : vector<8x64xbf16>, vector<4x64xbf16> into vector<4x8xf32>
//       CHECK:   %[[SA_BCAST:.+]] = vector.broadcast %[[SA]] : f32 to vector<4x8xf32>
//       CHECK:   %[[MUL_SA:.+]] = arith.mulf %[[PARTIAL]], %[[SA_BCAST]] : vector<4x8xf32>
//       CHECK:   %[[SB_BCAST:.+]] = vector.broadcast %[[SB]] : f32 to vector<4x8xf32>
//       CHECK:   %[[MUL_SB:.+]] = arith.mulf %[[MUL_SA]], %[[SB_BCAST]] : vector<4x8xf32>
//       CHECK:   %[[RES:.+]] = arith.addf %[[C]], %[[MUL_SB]] : vector<4x8xf32>
//       CHECK:   return %[[RES]]
func.func @scaled_contract_tensor_scaling(
    %A: vector<8x64xbf16>, %sA: f32, %B: vector<4x64xbf16>, %sB: f32,
    %C: vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> ()>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> ()>,
                     affine_map<(m, n, k) -> (n, m)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<8x64xbf16>, f32, vector<4x64xbf16>, f32 into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// Output narrower than f32 without reduction dims left after the contraction
// is rounded once by the only accumulation.

// CHECK-LABEL: func @scaled_contract_narrow_output_no_reduction
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<8x4xbf16>
//       CHECK:   vector.contract
//  CHECK-SAME:     : vector<8x64xf8E4M3FN>, vector<4x64xf8E4M3FN> into vector<8x4xf32>
//       CHECK:   arith.extf {{.+}} : vector<8xbf16> to vector<8xf32>
//       CHECK:   arith.extf {{.+}} : vector<4xbf16> to vector<4xf32>
//       CHECK:   %[[SCALED:.+]] = arith.mulf {{.+}} : vector<8x4xf32>
//       CHECK:   %[[C_EXT:.+]] = arith.extf %[[C]] : vector<8x4xbf16> to vector<8x4xf32>
//       CHECK:   %[[ACC:.+]] = arith.addf %[[C_EXT]], %[[SCALED]] : vector<8x4xf32>
//       CHECK:   %[[RES:.+]] = arith.truncf %[[ACC]] : vector<8x4xf32> to vector<8x4xbf16>
//       CHECK:   return %[[RES]]
func.func @scaled_contract_narrow_output_no_reduction(
    %A: vector<8x64xf8E4M3FN>, %sA: vector<8xbf16>,
    %B: vector<4x64xf8E4M3FN>, %sB: vector<4xbf16>,
    %C: vector<8x4xbf16>) -> vector<8x4xbf16> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<8x64xf8E4M3FN>, vector<8xbf16>,
      vector<4x64xf8E4M3FN>, vector<4xbf16> into vector<8x4xbf16>
  return %0 : vector<8x4xbf16>
}

// Output wider than f32 is accumulated in the output precision.

// CHECK-LABEL: func @scaled_contract_wide_output
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<8x4xf64>
//       CHECK:   vector.contract
//  CHECK-SAME:     : vector<8x2x32xf8E4M3FN>, vector<4x2x32xf8E4M3FN> into vector<8x4x2xf64>
//       CHECK:   arith.extf {{.+}} : vector<8x2xf8E8M0FNU> to vector<8x2xf64>
//       CHECK:   arith.extf {{.+}} : vector<4x2xf8E8M0FNU> to vector<4x2xf64>
//       CHECK:   %[[SCALED:.+]] = arith.mulf {{.+}} : vector<8x4x2xf64>
//       CHECK:   %[[RES:.+]] = vector.multi_reduction <add>, %[[SCALED]], %[[C]] [2] : vector<8x4x2xf64> to vector<8x4xf64>
//       CHECK:   return %[[RES]]
func.func @scaled_contract_wide_output(
    %A: vector<8x64xf8E4M3FN>, %sA: vector<8x2xf8E8M0FNU>,
    %B: vector<4x64xf8E4M3FN>, %sB: vector<4x2xf8E8M0FNU>,
    %C: vector<8x4xf64>) -> vector<8x4xf64> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 32)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 32)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<8x64xf8E4M3FN>, vector<8x2xf8E8M0FNU>,
      vector<4x64xf8E4M3FN>, vector<4x2xf8E8M0FNU> into vector<8x4xf64>
  return %0 : vector<8x4xf64>
}

// Dot product - the block results are reduced into a scalar accumulator.

// CHECK-LABEL: func @scaled_contract_to_scalar
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<64xf8E4M3FN>, %[[SA:[a-zA-Z0-9]+]]: vector<2xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<64xf8E4M3FN>, %[[SB:[a-zA-Z0-9]+]]: vector<2xf8E8M0FNU>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: f32
//       CHECK:   %[[A_BLK:.+]] = vector.shape_cast %[[A]] : vector<64xf8E4M3FN> to vector<2x32xf8E4M3FN>
//       CHECK:   %[[B_BLK:.+]] = vector.shape_cast %[[B]] : vector<64xf8E4M3FN> to vector<2x32xf8E4M3FN>
//       CHECK:   %[[PARTIAL:.+]] = vector.contract
//  CHECK-SAME:     indexing_maps = [#[[$MAP_DOT_IN]], #[[$MAP_DOT_IN]], #[[$MAP_DOT_PARTIAL]]]
//  CHECK-SAME:     iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:     %[[A_BLK]], %[[B_BLK]], %{{.+}} : vector<2x32xf8E4M3FN>, vector<2x32xf8E4M3FN> into vector<2xf32>
//       CHECK:   %[[SA_EXT:.+]] = arith.extf %[[SA]] : vector<2xf8E8M0FNU> to vector<2xf32>
//       CHECK:   %[[MUL_SA:.+]] = arith.mulf %[[PARTIAL]], %[[SA_EXT]] : vector<2xf32>
//       CHECK:   %[[SB_EXT:.+]] = arith.extf %[[SB]] : vector<2xf8E8M0FNU> to vector<2xf32>
//       CHECK:   %[[MUL_SB:.+]] = arith.mulf %[[MUL_SA]], %[[SB_EXT]] : vector<2xf32>
//       CHECK:   %[[RES:.+]] = vector.multi_reduction <add>, %[[MUL_SB]], %[[C]] [0] : vector<2xf32> to f32
//       CHECK:   return %[[RES]] : f32
func.func @scaled_contract_to_scalar(
    %A: vector<64xf8E4M3FN>, %sA: vector<2xf8E8M0FNU>,
    %B: vector<64xf8E4M3FN>, %sB: vector<2xf8E8M0FNU>,
    %C: f32) -> f32 {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(k) -> (k)>,
                     affine_map<(k) -> (k floordiv 32)>,
                     affine_map<(k) -> (k)>,
                     affine_map<(k) -> (k floordiv 32)>,
                     affine_map<(k) -> ()>],
    iterator_types = ["reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<64xf8E4M3FN>, vector<2xf8E8M0FNU>,
      vector<64xf8E4M3FN>, vector<2xf8E8M0FNU> into f32
  return %0 : f32
}

// Batch dim with transposed rhs - the transposed scale is permuted into the
// layout of the partial results.

// CHECK-LABEL: func @scaled_contract_batch
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<2x8x64xf8E4M3FN>, %[[SA:[a-zA-Z0-9]+]]: vector<2x8x2xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<2x64x4xf8E4M3FN>, %[[SB:[a-zA-Z0-9]+]]: vector<2x2x4xf8E8M0FNU>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<2x8x4xf32>
//       CHECK:   %[[A_BLK:.+]] = vector.shape_cast %[[A]] : vector<2x8x64xf8E4M3FN> to vector<2x8x2x32xf8E4M3FN>
//       CHECK:   %[[B_BLK:.+]] = vector.shape_cast %[[B]] : vector<2x64x4xf8E4M3FN> to vector<2x2x32x4xf8E4M3FN>
//       CHECK:   vector.contract
//  CHECK-SAME:     indexing_maps = [#[[$MAP_BATCH_LHS]], #[[$MAP_BATCH_RHS]], #[[$MAP_BATCH_PARTIAL]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:     %[[A_BLK]], %[[B_BLK]], %{{.+}} : vector<2x8x2x32xf8E4M3FN>, vector<2x2x32x4xf8E4M3FN> into vector<2x8x4x2xf32>
//       CHECK:   %[[SB_EXT:.+]] = arith.extf %[[SB]] : vector<2x2x4xf8E8M0FNU> to vector<2x2x4xf32>
//       CHECK:   %[[SB_T:.+]] = vector.transpose %[[SB_EXT]], [0, 2, 1] : vector<2x2x4xf32> to vector<2x4x2xf32>
//       CHECK:   %[[SB_CAST:.+]] = vector.shape_cast %[[SB_T]] : vector<2x4x2xf32> to vector<2x1x4x2xf32>
//       CHECK:   %[[SB_BCAST:.+]] = vector.broadcast %[[SB_CAST]] : vector<2x1x4x2xf32> to vector<2x8x4x2xf32>
//       CHECK:   %[[SCALED:.+]] = arith.mulf %{{.+}}, %[[SB_BCAST]] : vector<2x8x4x2xf32>
//       CHECK:   %[[RES:.+]] = vector.multi_reduction <add>, %[[SCALED]], %[[C]] [3] : vector<2x8x4x2xf32> to vector<2x8x4xf32>
//       CHECK:   return %[[RES]]
func.func @scaled_contract_batch(
    %A: vector<2x8x64xf8E4M3FN>, %sA: vector<2x8x2xf8E8M0FNU>,
    %B: vector<2x64x4xf8E4M3FN>, %sB: vector<2x2x4xf8E8M0FNU>,
    %C: vector<2x8x4xf32>) -> vector<2x8x4xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                     affine_map<(b, m, n, k) -> (b, m, k floordiv 32)>,
                     affine_map<(b, m, n, k) -> (b, k, n)>,
                     affine_map<(b, m, n, k) -> (b, k floordiv 32, n)>,
                     affine_map<(b, m, n, k) -> (b, m, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<2x8x64xf8E4M3FN>, vector<2x8x2xf8E8M0FNU>,
      vector<2x64x4xf8E4M3FN>, vector<2x2x4xf8E8M0FNU> into vector<2x8x4xf32>
  return %0 : vector<2x8x4xf32>
}

// Integers wider than 8 bits are contracted in the compute type as
// vector.contract only promotes integers to wider integers.

// CHECK-LABEL: func @scaled_contract_wide_int
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<4x32xi16>, %[[SA:[a-zA-Z0-9]+]]: vector<4xf32>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<2x32xi16>
//       CHECK:   %[[A_CVT:.+]] = arith.sitofp %[[A]] : vector<4x32xi16> to vector<4x32xf32>
//       CHECK:   %[[B_CVT:.+]] = arith.sitofp %[[B]] : vector<2x32xi16> to vector<2x32xf32>
//       CHECK:   vector.contract
//  CHECK-SAME:     %[[A_CVT]], %[[B_CVT]], %{{.+}} : vector<4x32xf32>, vector<2x32xf32> into vector<4x2xf32>
func.func @scaled_contract_wide_int(
    %A: vector<4x32xi16>, %sA: vector<4xf32>,
    %B: vector<2x32xi16>, %sB: vector<2xf32>,
    %C: vector<4x2xf32>) -> vector<4x2xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<4x32xi16>, vector<4xf32>,
      vector<2x32xi16>, vector<2xf32> into vector<4x2xf32>
  return %0 : vector<4x2xf32>
}

// Dot product with output narrower than f32 - the block results are reduced
// in f32 and rounded once.

// CHECK-LABEL: func @scaled_contract_to_scalar_narrow_output
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: f16
//       CHECK:   %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   arith.extf {{.+}} : vector<2xf8E8M0FNU> to vector<2xf32>
//       CHECK:   arith.extf {{.+}} : vector<2xf8E8M0FNU> to vector<2xf32>
//       CHECK:   %[[SCALED:.+]] = arith.mulf {{.+}} : vector<2xf32>
//       CHECK:   %[[SUM:.+]] = vector.multi_reduction <add>, %[[SCALED]], %[[ZERO]] [0] : vector<2xf32> to f32
//       CHECK:   %[[C_EXT:.+]] = arith.extf %[[C]] : f16 to f32
//       CHECK:   %[[ACC:.+]] = arith.addf %[[C_EXT]], %[[SUM]] : f32
//       CHECK:   %[[RES:.+]] = arith.truncf %[[ACC]] : f32 to f16
//       CHECK:   return %[[RES]] : f16
func.func @scaled_contract_to_scalar_narrow_output(
    %A: vector<64xf8E4M3FN>, %sA: vector<2xf8E8M0FNU>,
    %B: vector<64xf8E4M3FN>, %sB: vector<2xf8E8M0FNU>,
    %C: f16) -> f16 {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(k) -> (k)>,
                     affine_map<(k) -> (k floordiv 32)>,
                     affine_map<(k) -> (k)>,
                     affine_map<(k) -> (k floordiv 32)>,
                     affine_map<(k) -> ()>],
    iterator_types = ["reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<64xf8E4M3FN>, vector<2xf8E8M0FNU>,
      vector<64xf8E4M3FN>, vector<2xf8E8M0FNU> into f16
  return %0 : f16
}

// Block scaling of a parallel dim with transposed accumulator - the split dim
// is reshaped in the accumulator layout.

// CHECK-LABEL: func @scaled_contract_parallel_block_transposed_acc
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<4x64xf8E4M3FN>, %[[SA:[a-zA-Z0-9]+]]: vector<2x2xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<2x64xf8E4M3FN>, %[[SB:[a-zA-Z0-9]+]]: f32
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<2x4xf32>
//       CHECK:   %[[A_BLK:.+]] = vector.shape_cast %[[A]] : vector<4x64xf8E4M3FN> to vector<2x2x2x32xf8E4M3FN>
//       CHECK:   %[[B_BLK:.+]] = vector.shape_cast %[[B]] : vector<2x64xf8E4M3FN> to vector<2x2x32xf8E4M3FN>
//       CHECK:   %[[PARTIAL:.+]] = vector.contract
//  CHECK-SAME:     %[[A_BLK]], %[[B_BLK]], %{{.+}} : vector<2x2x2x32xf8E4M3FN>, vector<2x2x32xf8E4M3FN> into vector<2x2x2x2xf32>
//       CHECK:   %[[SA_EXT:.+]] = arith.extf %[[SA]] : vector<2x2xf8E8M0FNU> to vector<2x2xf32>
//       CHECK:   %[[SA_CAST:.+]] = vector.shape_cast %[[SA_EXT]] : vector<2x2xf32> to vector<1x2x1x2xf32>
//       CHECK:   vector.broadcast %[[SA_CAST]] : vector<1x2x1x2xf32> to vector<2x2x2x2xf32>
//       CHECK:   vector.broadcast %[[SB]] : f32 to vector<2x2x2x2xf32>
//       CHECK:   %[[SCALED:.+]] = arith.mulf {{.+}} : vector<2x2x2x2xf32>
//       CHECK:   %[[C_BLK:.+]] = vector.shape_cast %[[C]] : vector<2x4xf32> to vector<2x2x2xf32>
//       CHECK:   %[[SUM:.+]] = vector.multi_reduction <add>, %[[SCALED]], %[[C_BLK]] [3] : vector<2x2x2x2xf32> to vector<2x2x2xf32>
//       CHECK:   %[[RES:.+]] = vector.shape_cast %[[SUM]] : vector<2x2x2xf32> to vector<2x4xf32>
//       CHECK:   return %[[RES]]
func.func @scaled_contract_parallel_block_transposed_acc(
    %A: vector<4x64xf8E4M3FN>, %sA: vector<2x2xf8E8M0FNU>,
    %B: vector<2x64xf8E4M3FN>, %sB: f32,
    %C: vector<2x4xf32>) -> vector<2x4xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m floordiv 2, k floordiv 32)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> ()>,
                     affine_map<(m, n, k) -> (n, m)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<4x64xf8E4M3FN>, vector<2x2xf8E8M0FNU>,
      vector<2x64xf8E4M3FN>, f32 into vector<2x4xf32>
  return %0 : vector<2x4xf32>
}

//===----------------------------------------------------------------------===//
// [Pattern: DecomposeScaledContractByPrescaling]
//===----------------------------------------------------------------------===//

// Scaling of every element along the reduction dim leaves nothing to contract
// within blocks, so the inputs are scaled before the contraction.

// CHECK-LABEL: func @scaled_contract_reduction_dim_scaling
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<8x64xf8E4M3FN>, %[[SA:[a-zA-Z0-9]+]]: vector<8x64xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<64x4xf8E4M3FN>, %[[SB:[a-zA-Z0-9]+]]: vector<2x4xf8E8M0FNU>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<8x4xf16>
//       CHECK:   %[[A_EXT:.+]] = arith.extf %[[A]] : vector<8x64xf8E4M3FN> to vector<8x64xf32>
//       CHECK:   %[[SA_EXT:.+]] = arith.extf %[[SA]] : vector<8x64xf8E8M0FNU> to vector<8x64xf32>
//       CHECK:   %[[A_SCALED:.+]] = arith.mulf %[[A_EXT]], %[[SA_EXT]] : vector<8x64xf32>
//       CHECK:   %[[B_EXT:.+]] = arith.extf %[[B]] : vector<64x4xf8E4M3FN> to vector<64x4xf32>
//       CHECK:   %[[SB_EXT:.+]] = arith.extf %[[SB]] : vector<2x4xf8E8M0FNU> to vector<2x4xf32>
//       CHECK:   %[[SB_CAST:.+]] = vector.shape_cast %[[SB_EXT]] : vector<2x4xf32> to vector<2x1x4xf32>
//       CHECK:   %[[SB_BCAST:.+]] = vector.broadcast %[[SB_CAST]] : vector<2x1x4xf32> to vector<2x32x4xf32>
//       CHECK:   %[[SB_EXPAND:.+]] = vector.shape_cast %[[SB_BCAST]] : vector<2x32x4xf32> to vector<64x4xf32>
//       CHECK:   %[[B_SCALED:.+]] = arith.mulf %[[B_EXT]], %[[SB_EXPAND]] : vector<64x4xf32>
//       CHECK:   %[[C_EXT:.+]] = arith.extf %[[C]] : vector<8x4xf16> to vector<8x4xf32>
//       CHECK:   %[[CONTRACT:.+]] = vector.contract
//  CHECK-SAME:     indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS_T]], #[[$MAP_ACC]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     %[[A_SCALED]], %[[B_SCALED]], %[[C_EXT]] : vector<8x64xf32>, vector<64x4xf32> into vector<8x4xf32>
//       CHECK:   %[[RES:.+]] = arith.truncf %[[CONTRACT]] : vector<8x4xf32> to vector<8x4xf16>
//       CHECK:   return %[[RES]]
func.func @scaled_contract_reduction_dim_scaling(
    %A: vector<8x64xf8E4M3FN>, %sA: vector<8x64xf8E8M0FNU>,
    %B: vector<64x4xf8E4M3FN>, %sB: vector<2x4xf8E8M0FNU>,
    %C: vector<8x4xf16>) -> vector<8x4xf16> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (k, n)>,
                     affine_map<(m, n, k) -> (k floordiv 32, n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<8x64xf8E4M3FN>, vector<8x64xf8E8M0FNU>,
      vector<64x4xf8E4M3FN>, vector<2x4xf8E8M0FNU> into vector<8x4xf16>
  return %0 : vector<8x4xf16>
}

// Block sizes of the reduction dim which do not divide each other cannot be
// split into block levels.

// CHECK-LABEL: func @scaled_contract_incompatible_blocks
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<8x96xi8>, %[[SA:[a-zA-Z0-9]+]]: vector<8x3xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<4x96xi8>, %[[SB:[a-zA-Z0-9]+]]: vector<2xf32>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<8x4xf32>
//       CHECK:   %[[A_CVT:.+]] = arith.sitofp %[[A]] : vector<8x96xi8> to vector<8x96xf32>
//       CHECK:   %[[SA_EXT:.+]] = arith.extf %[[SA]] : vector<8x3xf8E8M0FNU> to vector<8x3xf32>
//       CHECK:   %[[SA_CAST:.+]] = vector.shape_cast %[[SA_EXT]] : vector<8x3xf32> to vector<8x3x1xf32>
//       CHECK:   %[[SA_BCAST:.+]] = vector.broadcast %[[SA_CAST]] : vector<8x3x1xf32> to vector<8x3x32xf32>
//       CHECK:   %[[SA_EXPAND:.+]] = vector.shape_cast %[[SA_BCAST]] : vector<8x3x32xf32> to vector<8x96xf32>
//       CHECK:   %[[A_SCALED:.+]] = arith.mulf %[[A_CVT]], %[[SA_EXPAND]] : vector<8x96xf32>
//       CHECK:   %[[B_CVT:.+]] = arith.sitofp %[[B]] : vector<4x96xi8> to vector<4x96xf32>
//       CHECK:   %[[SB_CAST:.+]] = vector.shape_cast %[[SB]] : vector<2xf32> to vector<1x2x1xf32>
//       CHECK:   %[[SB_BCAST:.+]] = vector.broadcast %[[SB_CAST]] : vector<1x2x1xf32> to vector<4x2x48xf32>
//       CHECK:   %[[SB_EXPAND:.+]] = vector.shape_cast %[[SB_BCAST]] : vector<4x2x48xf32> to vector<4x96xf32>
//       CHECK:   %[[B_SCALED:.+]] = arith.mulf %[[B_CVT]], %[[SB_EXPAND]] : vector<4x96xf32>
//       CHECK:   %[[RES:.+]] = vector.contract
//  CHECK-SAME:     %[[A_SCALED]], %[[B_SCALED]], %[[C]] : vector<8x96xf32>, vector<4x96xf32> into vector<8x4xf32>
//       CHECK:   return %[[RES]]
func.func @scaled_contract_incompatible_blocks(
    %A: vector<8x96xi8>, %sA: vector<8x3xf8E8M0FNU>,
    %B: vector<4x96xi8>, %sB: vector<2xf32>,
    %C: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (m, k floordiv 32)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (k floordiv 48)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<8x96xi8>, vector<8x3xf8E8M0FNU>,
      vector<4x96xi8>, vector<2xf32> into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// Block sizes of a batch dim which do not divide each other cannot be split
// into block levels.

// CHECK-LABEL: func @scaled_contract_incompatible_parallel_blocks
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<6x8x64xf8E4M3FN>, %[[SA:[a-zA-Z0-9]+]]: vector<3x8xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<6x4x64xf8E4M3FN>, %[[SB:[a-zA-Z0-9]+]]: vector<2x4xf8E8M0FNU>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: vector<6x8x4xf32>
//       CHECK:   %[[SA_EXT:.+]] = arith.extf %[[SA]] : vector<3x8xf8E8M0FNU> to vector<3x8xf32>
//       CHECK:   %[[SA_CAST:.+]] = vector.shape_cast %[[SA_EXT]] : vector<3x8xf32> to vector<3x1x8x1xf32>
//       CHECK:   %[[SA_BCAST:.+]] = vector.broadcast %[[SA_CAST]] : vector<3x1x8x1xf32> to vector<3x2x8x64xf32>
//       CHECK:   vector.shape_cast %[[SA_BCAST]] : vector<3x2x8x64xf32> to vector<6x8x64xf32>
//       CHECK:   %[[SB_EXT:.+]] = arith.extf %[[SB]] : vector<2x4xf8E8M0FNU> to vector<2x4xf32>
//       CHECK:   %[[SB_CAST:.+]] = vector.shape_cast %[[SB_EXT]] : vector<2x4xf32> to vector<2x1x4x1xf32>
//       CHECK:   %[[SB_BCAST:.+]] = vector.broadcast %[[SB_CAST]] : vector<2x1x4x1xf32> to vector<2x3x4x64xf32>
//       CHECK:   vector.shape_cast %[[SB_BCAST]] : vector<2x3x4x64xf32> to vector<6x4x64xf32>
//       CHECK:   %[[RES:.+]] = vector.contract
//  CHECK-SAME:     indexing_maps = [#[[$MAP_PRESCALE_BATCH_LHS]], #[[$MAP_BLK_LHS]], #[[$MAP_BLK_PARTIAL]]]
//  CHECK-SAME:     : vector<6x8x64xf32>, vector<6x4x64xf32> into vector<6x8x4xf32>
//       CHECK:   return %[[RES]]
func.func @scaled_contract_incompatible_parallel_blocks(
    %A: vector<6x8x64xf8E4M3FN>, %sA: vector<3x8xf8E8M0FNU>,
    %B: vector<6x4x64xf8E4M3FN>, %sB: vector<2x4xf8E8M0FNU>,
    %C: vector<6x8x4xf32>) -> vector<6x8x4xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                     affine_map<(b, m, n, k) -> (b floordiv 2, m)>,
                     affine_map<(b, m, n, k) -> (b, n, k)>,
                     affine_map<(b, m, n, k) -> (b floordiv 3, n)>,
                     affine_map<(b, m, n, k) -> (b, m, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<6x8x64xf8E4M3FN>, vector<3x8xf8E8M0FNU>,
      vector<6x4x64xf8E4M3FN>, vector<2x4xf8E8M0FNU> into vector<6x8x4xf32>
  return %0 : vector<6x8x4xf32>
}

// Transposed scale is permuted into the data layout and a single block
// spanning the whole dim is broadcast.

// CHECK-LABEL: func @scaled_contract_prescale_transposed_single_block
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: vector<4x16xf8E4M3FN>, %[[SA:[a-zA-Z0-9]+]]: vector<16x4xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: vector<2x16xf8E4M3FN>, %[[SB:[a-zA-Z0-9]+]]: vector<2x1xf8E8M0FNU>
//       CHECK:   %[[SA_EXT:.+]] = arith.extf %[[SA]] : vector<16x4xf8E8M0FNU> to vector<16x4xf32>
//       CHECK:   vector.transpose %[[SA_EXT]], [1, 0] : vector<16x4xf32> to vector<4x16xf32>
//       CHECK:   %[[SB_EXT:.+]] = arith.extf %[[SB]] : vector<2x1xf8E8M0FNU> to vector<2x1xf32>
//       CHECK:   vector.broadcast %[[SB_EXT]] : vector<2x1xf32> to vector<2x16xf32>
//       CHECK:   vector.contract
//  CHECK-SAME:     : vector<4x16xf32>, vector<2x16xf32> into vector<4x2xf32>
func.func @scaled_contract_prescale_transposed_single_block(
    %A: vector<4x16xf8E4M3FN>, %sA: vector<16x4xf8E8M0FNU>,
    %B: vector<2x16xf8E4M3FN>, %sB: vector<2x1xf8E8M0FNU>,
    %C: vector<4x2xf32>) -> vector<4x2xf32> {
  %0 = vector.scaled_contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (k, m)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (n, k floordiv 32)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    %A, %sA, %B, %sB, %C
    : vector<4x16xf8E4M3FN>, vector<16x4xf8E8M0FNU>,
      vector<2x16xf8E4M3FN>, vector<2x1xf8E8M0FNU> into vector<4x2xf32>
  return %0 : vector<4x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.decompose_scaled_contract
    } : !transform.any_op
    transform.yield
  }
}
