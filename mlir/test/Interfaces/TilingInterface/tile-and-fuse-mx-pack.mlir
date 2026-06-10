// RUN: mlir-opt %s --transform-interpreter --cse -split-input-file -verify-diagnostics | FileCheck %s

// Fuse `linalg.mx_pack` as a consumer of a tiled producer loop. The producer
// (`linalg.negf`) is tiled along dim 0 by 8 (one scaling block). The mx_pack
// consumer is pulled into the loop: the `source` tile maps directly onto the
// iteration domain, the `dest` tile follows it, and the `scale_dest` tile is
// inferred via floordiv by the block size.
func.func @fuse_mx_pack_consumer(%src: tensor<128x768xf16>,
    %data: tensor<128x768xi8>, %scales: tensor<16x24xf8E8M0FNU>)
    -> (tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %empty = tensor.empty() : tensor<128x768xf16>
  %0 = scf.for %i = %c0 to %c128 step %c8 iter_args(%it0 = %empty) -> (tensor<128x768xf16>) {
    %s = tensor.extract_slice %src[%i, 0] [8, 768] [1, 1]
      : tensor<128x768xf16> to tensor<8x768xf16>
    %e = tensor.extract_slice %it0[%i, 0] [8, 768] [1, 1]
      : tensor<128x768xf16> to tensor<8x768xf16>
    %neg = linalg.negf ins(%s : tensor<8x768xf16>) outs(%e : tensor<8x768xf16>) -> tensor<8x768xf16>
    %ins = tensor.insert_slice %neg into %it0[%i, 0] [8, 768] [1, 1]
      : tensor<8x768xf16> into tensor<128x768xf16>
    scf.yield %ins : tensor<128x768xf16>
  }
  %pack:2 = linalg.mx_pack %0
    scale_dims = [0, 1]
    scale_blocks = [8, 32]
    into %data, %scales
    : tensor<128x768xf16> -> tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
  return %pack#0, %pack#1 : tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %pack = transform.structured.match ops{["linalg.mx_pack"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %new_loop = transform.test.fuse_consumer %pack into (%loop)
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-LABEL: func.func @fuse_mx_pack_consumer(
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: tensor<128x768xf16>
//  CHECK-SAME:     %[[DATA:[a-zA-Z0-9]+]]: tensor<128x768xi8>
//  CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<16x24xf8E8M0FNU>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x768xf16>
//       CHECK:   %[[LOOP:.+]]:3 = scf.for %[[IV:[a-zA-Z0-9]+]] =
//  CHECK-SAME:       iter_args(%[[IT_SRC:.+]] = %[[EMPTY]], %[[IT_DATA:.+]] = %[[DATA]], %[[IT_SCALE:.+]] = %[[SCALES]])
//       CHECK:     %[[NEG:.+]] = linalg.negf
//       CHECK:     tensor.insert_slice %[[NEG]] into %[[IT_SRC]][%[[IV]], 0] [8, 768] [1, 1]
//   CHECK-DAG:     %[[SOFF:.+]] = affine.apply #[[$MAP0]](%[[IV]])
//   CHECK-DAG:     %[[DATA_TILE:.+]] = tensor.extract_slice %[[IT_DATA]][%[[IV]], 0] [8, 768] [1, 1]
//   CHECK-DAG:     %[[SCALE_TILE:.+]] = tensor.extract_slice %[[IT_SCALE]][%[[SOFF]], 0] [1, 24] [1, 1]
//       CHECK:     %[[PACK_DATA:.+]], %[[PACK_SCALE:.+]] = linalg.mx_pack %[[NEG]]
//  CHECK-SAME:         scale_dims = [0, 1] scale_blocks = [8, 32]
//  CHECK-SAME:         into %[[DATA_TILE]], %[[SCALE_TILE]]
//       CHECK:     %[[INS_DATA:.+]] = tensor.insert_slice %[[PACK_DATA]] into %[[IT_DATA]][%[[IV]], 0] [8, 768] [1, 1]
//       CHECK:     %[[INS_SCALE:.+]] = tensor.insert_slice %[[PACK_SCALE]] into %[[IT_SCALE]][%[[SOFF]], 0] [1, 24] [1, 1]
//       CHECK:     scf.yield %{{.+}}, %[[INS_DATA]], %[[INS_SCALE]]
//       CHECK:   return %[[LOOP]]#1, %[[LOOP]]#2

// -----

// The same tiling constraints apply during consumer fusion. Here the producer
// (`linalg.negf`) is tiled along the scaling dim 0 by 4, which is not a
// multiple of the scaling block (8). Fusing the mx_pack consumer would require
// a source tile of size 4, which splits a scaling block, so the fusion fails.
func.func @no_fuse_mx_pack_consumer_partial_block(%src: tensor<128x768xf16>,
    %data: tensor<128x768xi8>, %scales: tensor<16x24xf8E8M0FNU>)
    -> (tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %empty = tensor.empty() : tensor<128x768xf16>
  %0 = scf.for %i = %c0 to %c128 step %c4 iter_args(%it0 = %empty) -> (tensor<128x768xf16>) {
    %s = tensor.extract_slice %src[%i, 0] [4, 768] [1, 1]
      : tensor<128x768xf16> to tensor<4x768xf16>
    %e = tensor.extract_slice %it0[%i, 0] [4, 768] [1, 1]
      : tensor<128x768xf16> to tensor<4x768xf16>
    %neg = linalg.negf ins(%s : tensor<4x768xf16>) outs(%e : tensor<4x768xf16>) -> tensor<4x768xf16>
    %ins = tensor.insert_slice %neg into %it0[%i, 0] [4, 768] [1, 1]
      : tensor<4x768xf16> into tensor<128x768xf16>
    scf.yield %ins : tensor<128x768xf16>
  }
  // expected-error @below {{failed to fuse consumer of slice}}
  %pack:2 = linalg.mx_pack %0
    scale_dims = [0, 1]
    scale_blocks = [8, 32]
    into %data, %scales
    : tensor<128x768xf16> -> tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
  return %pack#0, %pack#1 : tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %pack = transform.structured.match ops{["linalg.mx_pack"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %new_loop = transform.test.fuse_consumer %pack into (%loop)
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Fuse `linalg.mx_pack` as a producer into a consumer (`linalg.copy`) that
// reads the packed `dest` result. The consumer is tiled along dim 0 by 8, and
// the mx_pack producing the tile is generated inside the loop (its `scale_dest`
// tile is inferred via floordiv by the block size).
func.func @fuse_mx_pack_producer(%src: tensor<128x768xf16>,
    %data: tensor<128x768xi8>, %scales: tensor<16x24xf8E8M0FNU>,
    %out: tensor<128x768xi8>)
    -> (tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>) {
  %pack:2 = linalg.mx_pack %src
    scale_dims = [0, 1]
    scale_blocks = [8, 32]
    into %data, %scales
    : tensor<128x768xf16> -> tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
  %copy = linalg.copy ins(%pack#0 : tensor<128x768xi8>) outs(%out : tensor<128x768xi8>) -> tensor<128x768xi8>
  return %copy, %pack#1 : tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %copy = transform.structured.match ops{["linalg.copy"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %loop = transform.structured.fuse %copy tile_sizes [8, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-LABEL: func.func @fuse_mx_pack_producer(
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: tensor<128x768xf16>
//  CHECK-SAME:     %[[DATA:[a-zA-Z0-9]+]]: tensor<128x768xi8>
//  CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<16x24xf8E8M0FNU>
//  CHECK-SAME:     %[[OUT:[a-zA-Z0-9]+]]: tensor<128x768xi8>
//       CHECK:   %[[LOOP:.+]] = scf.for %[[IV:[a-zA-Z0-9]+]] =
//  CHECK-SAME:       iter_args(%[[ITER:.+]] = %[[OUT]])
//   CHECK-DAG:     %[[SRC_TILE:.+]] = tensor.extract_slice %[[SRC]][%[[IV]], 0] [8, 768] [1, 1]
//   CHECK-DAG:     %[[DATA_TILE:.+]] = tensor.extract_slice %[[DATA]][%[[IV]], 0] [8, 768] [1, 1]
//   CHECK-DAG:     %[[SOFF:.+]] = affine.apply #[[$MAP0]](%[[IV]])
//   CHECK-DAG:     %[[SCALE_TILE:.+]] = tensor.extract_slice %[[SCALES]][%[[SOFF]], 0] [1, 24] [1, 1]
//       CHECK:     %[[PACK_DATA:.+]], %[[PACK_SCALE:.+]] = linalg.mx_pack %[[SRC_TILE]]
//  CHECK-SAME:         into %[[DATA_TILE]], %[[SCALE_TILE]]
//       CHECK:     %[[OUT_TILE:.+]] = tensor.extract_slice %[[ITER]][%[[IV]], 0] [8, 768] [1, 1]
//       CHECK:     %[[COPY:.+]] = linalg.copy ins(%[[PACK_DATA]] : tensor<8x768xi8>) outs(%[[OUT_TILE]]
//       CHECK:     %[[INS:.+]] = tensor.insert_slice %[[COPY]] into %[[ITER]][%[[IV]], 0] [8, 768] [1, 1]
//       CHECK:     scf.yield %[[INS]]

// -----

// Fuse `linalg.mx_pack` as a producer into a consumer that reads the
// `scale_dest` result. Tiling the scales consumer along dim 0 by 1 is inverted
// back to the data iteration domain: the source/data tile offset is `iv * 8`.
// Because dim 0 (128) is a static multiple of the block (8), every block is
// full, so the recovered data tile size is the static `8` (no clamping
// `affine.min`) and the scale tile keeps its static `1` size (no `ceildiv`).
// The fused `mx_pack` therefore has fully static shapes.
func.func @fuse_mx_pack_scales(%src: tensor<128x768xf16>,
    %data: tensor<128x768xi8>, %scales: tensor<16x24xf8E8M0FNU>,
    %out: tensor<16x24xf8E8M0FNU>)
    -> (tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>) {
  %pack:2 = linalg.mx_pack %src
    scale_dims = [0, 1]
    scale_blocks = [8, 32]
    into %data, %scales
    : tensor<128x768xf16> -> tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
  %copy = linalg.copy ins(%pack#1 : tensor<16x24xf8E8M0FNU>) outs(%out : tensor<16x24xf8E8M0FNU>) -> tensor<16x24xf8E8M0FNU>
  return %pack#0, %copy : tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %copy = transform.structured.match ops{["linalg.copy"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %loop = transform.structured.fuse %copy tile_sizes [1, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-LABEL: func.func @fuse_mx_pack_scales(
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: tensor<128x768xf16>
//  CHECK-SAME:     %[[DATA:[a-zA-Z0-9]+]]: tensor<128x768xi8>
//  CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<16x24xf8E8M0FNU>
//  CHECK-SAME:     %[[OUT:[a-zA-Z0-9]+]]: tensor<16x24xf8E8M0FNU>
//       CHECK:   %[[LOOP:.+]] = scf.for %[[IV:[a-zA-Z0-9]+]] =
//  CHECK-SAME:       iter_args(%[[ITER:.+]] = %[[OUT]])
//   CHECK-DAG:     %[[OFF:.+]] = affine.apply #[[$MAP0]](%[[IV]])
//   CHECK-DAG:     %[[SRC_TILE:.+]] = tensor.extract_slice %[[SRC]][%[[OFF]], 0] [8, 768] [1, 1]
//   CHECK-DAG:     %[[DATA_TILE:.+]] = tensor.extract_slice %[[DATA]][%[[OFF]], 0] [8, 768] [1, 1]
//   CHECK-DAG:     %[[SCALE_TILE:.+]] = tensor.extract_slice %[[SCALES]][%[[IV]], 0] [1, 24] [1, 1]
//       CHECK:     %[[PACK_DATA:.+]], %[[PACK_SCALE:.+]] = linalg.mx_pack %[[SRC_TILE]]
//  CHECK-SAME:         into %[[DATA_TILE]], %[[SCALE_TILE]]
//  CHECK-SAME:         -> tensor<8x768xi8>, tensor<1x24xf8E8M0FNU>
//       CHECK:     %[[OUT_TILE:.+]] = tensor.extract_slice %[[ITER]][%[[IV]], 0] [1, 24] [1, 1]
//       CHECK:     %[[COPY:.+]] = linalg.copy ins(%[[PACK_SCALE]] : tensor<1x24xf8E8M0FNU>) outs(%[[OUT_TILE]]
//       CHECK:     tensor.insert_slice %[[COPY]] into %[[ITER]][%[[IV]], 0] [1, 24] [1, 1]

// -----

// Counterpart to @fuse_mx_pack_scales where the scaling dimension (130) is not
// a static multiple of the block (8). The trailing block is partial, so the
// recovered data tile size cannot be the static `8`: it is clamped to the
// dimension with `affine.min` and the scale tile size is recomputed via
// `ceildiv`, yielding dynamic shapes for the fused `mx_pack`.
func.func @fuse_mx_pack_scales_dynamic(%src: tensor<130x768xf16>,
    %data: tensor<130x768xi8>, %scales: tensor<17x24xf8E8M0FNU>,
    %out: tensor<17x24xf8E8M0FNU>)
    -> (tensor<130x768xi8>, tensor<17x24xf8E8M0FNU>) {
  %pack:2 = linalg.mx_pack %src
    scale_dims = [0, 1]
    scale_blocks = [8, 32]
    into %data, %scales
    : tensor<130x768xf16> -> tensor<130x768xi8>, tensor<17x24xf8E8M0FNU>
  %copy = linalg.copy ins(%pack#1 : tensor<17x24xf8E8M0FNU>) outs(%out : tensor<17x24xf8E8M0FNU>) -> tensor<17x24xf8E8M0FNU>
  return %pack#0, %copy : tensor<130x768xi8>, tensor<17x24xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %copy = transform.structured.match ops{["linalg.copy"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %loop = transform.structured.fuse %copy tile_sizes [1, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//   CHECK-DAG: #[[$MUL:.+]] = affine_map<(d0) -> (d0 * 8)>
//   CHECK-DAG: #[[$MIN:.+]] = affine_map<(d0) -> (8, d0 * -8 + 130)>
//   CHECK-DAG: #[[$CEIL:.+]] = affine_map<(d0) -> (d0 ceildiv 8)>
// CHECK-LABEL: func.func @fuse_mx_pack_scales_dynamic(
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: tensor<130x768xf16>
//  CHECK-SAME:     %[[DATA:[a-zA-Z0-9]+]]: tensor<130x768xi8>
//  CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<17x24xf8E8M0FNU>
//  CHECK-SAME:     %[[OUT:[a-zA-Z0-9]+]]: tensor<17x24xf8E8M0FNU>
//       CHECK:   %[[LOOP:.+]] = scf.for %[[IV:[a-zA-Z0-9]+]] =
//  CHECK-SAME:       iter_args(%[[ITER:.+]] = %[[OUT]])
//   CHECK-DAG:     %[[OFF:.+]] = affine.apply #[[$MUL]](%[[IV]])
//   CHECK-DAG:     %[[SZ:.+]] = affine.min #[[$MIN]](%[[IV]])
//   CHECK-DAG:     %[[SRC_TILE:.+]] = tensor.extract_slice %[[SRC]][%[[OFF]], 0] [%[[SZ]], 768] [1, 1]
//   CHECK-DAG:     %[[DATA_TILE:.+]] = tensor.extract_slice %[[DATA]][%[[OFF]], 0] [%[[SZ]], 768] [1, 1]
//   CHECK-DAG:     %[[SCALE_SZ:.+]] = affine.apply #[[$CEIL]](%[[SZ]])
//   CHECK-DAG:     %[[SCALE_TILE:.+]] = tensor.extract_slice %[[SCALES]][%[[IV]], 0] [%[[SCALE_SZ]], 24] [1, 1]
//       CHECK:     %[[PACK_DATA:.+]], %[[PACK_SCALE:.+]] = linalg.mx_pack %[[SRC_TILE]]
//  CHECK-SAME:         into %[[DATA_TILE]], %[[SCALE_TILE]]
//  CHECK-SAME:         : tensor<?x768xf16> -> tensor<?x768xi8>, tensor<?x24xf8E8M0FNU>
//       CHECK:     %[[OUT_TILE:.+]] = tensor.extract_slice %[[ITER]][%[[IV]], 0] [1, 24] [1, 1]
//       CHECK:     %[[COPY:.+]] = linalg.copy ins(%[[PACK_SCALE]] : tensor<?x24xf8E8M0FNU>) outs(%[[OUT_TILE]]
//       CHECK:     tensor.insert_slice %[[COPY]] into %[[ITER]][%[[IV]], 0] [1, 24] [1, 1]

// -----

// The same tiling constraints apply during fusion: a producer tile whose size
// is not a multiple of the scaling block cannot be fused. Tiling the consumer
// dim 0 by 4 would make the mx_pack source tile 4 (not a multiple of block 8),
// so the producer is left un-fused (the full mx_pack stays outside the loop and
// the loop reads a slice of its result).
func.func @no_fuse_mx_pack_producer_partial_block(%src: tensor<128x768xf16>,
    %data: tensor<128x768xi8>, %scales: tensor<16x24xf8E8M0FNU>,
    %out: tensor<128x768xi8>)
    -> (tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>) {
  %pack:2 = linalg.mx_pack %src
    scale_dims = [0, 1]
    scale_blocks = [8, 32]
    into %data, %scales
    : tensor<128x768xf16> -> tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
  %copy = linalg.copy ins(%pack#0 : tensor<128x768xi8>) outs(%out : tensor<128x768xi8>) -> tensor<128x768xi8>
  return %copy, %pack#1 : tensor<128x768xi8>, tensor<16x24xf8E8M0FNU>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %copy = transform.structured.match ops{["linalg.copy"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %loop = transform.structured.fuse %copy tile_sizes [4, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func.func @no_fuse_mx_pack_producer_partial_block(
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: tensor<128x768xf16>
//  CHECK-SAME:     %[[DATA:[a-zA-Z0-9]+]]: tensor<128x768xi8>
//  CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<16x24xf8E8M0FNU>
//  CHECK-SAME:     %[[OUT:[a-zA-Z0-9]+]]: tensor<128x768xi8>
//       CHECK:   %[[PACK_DATA:.+]], %[[PACK_SCALE:.+]] = linalg.mx_pack %[[SRC]]
//  CHECK-SAME:       into %[[DATA]], %[[SCALES]]
//       CHECK:   %[[LOOP:.+]] = scf.for %[[IV:[a-zA-Z0-9]+]] =
//  CHECK-SAME:       iter_args(%[[ITER:.+]] = %[[OUT]])
//   CHECK-NOT:     linalg.mx_pack
//       CHECK:     %[[DATA_TILE:.+]] = tensor.extract_slice %[[PACK_DATA]][%[[IV]], 0] [4, 768] [1, 1]
//       CHECK:     %[[COPY:.+]] = linalg.copy ins(%[[DATA_TILE]]
//       CHECK:     tensor.insert_slice %[[COPY]] into %[[ITER]][%[[IV]], 0] [4, 768] [1, 1]
