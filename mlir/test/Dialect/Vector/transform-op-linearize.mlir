// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func @linearize_elementwise
//  CHECK-SAME:   %[[ARG:.*]]: vector<32xf32>
//       CHECK:   %[[TRUNC:.*]] = arith.truncf %[[ARG]] : vector<32xf32> to vector<32xbf16>
//       CHECK:   return %[[TRUNC]] : vector<32xbf16>
func.func @linearize_elementwise(%arg0: vector<32xf32>) -> vector<32xbf16> {
  %0 = vector.shape_cast %arg0 : vector<32xf32> to vector<16x2xf32>
  %1 = arith.truncf %0 : vector<16x2xf32> to vector<16x2xbf16>
  %2 = vector.shape_cast %1 : vector<16x2xbf16> to vector<32xbf16>
  return %2 : vector<32xbf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %f {
      transform.apply_conversion_patterns.vector.linearize
    } <partial_conversion> : !transform.any_op
    %g = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %g {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @linearize_shuffle_like
//       CHECK:   vector.shuffle
//   CHECK-NOT:   vector.extract_strided_slice
func.func @linearize_shuffle_like(%arg0: vector<4x8xf32>) -> vector<2x8xf32> {
  %0 = vector.extract_strided_slice %arg0 offsets = [1, 0], sizes = [2, 8], strides = [1, 1]
    : vector<4x8xf32> to vector<2x8xf32>
  return %0 : vector<2x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %f {
      transform.apply_conversion_patterns.vector.linearize shuffle_like_ops = true
    } <partial_conversion> : !transform.any_op
    transform.yield
  }
}
