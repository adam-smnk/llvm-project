// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

// CHECK-LABEL: func @exp_vector
// CHECK-NOT:     math.exp
// CHECK:         math.fma
func.func @exp_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.exp %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.math.polynomial_approximation
    } : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @only_tanh
// CHECK-NOT:     math.tanh
// CHECK:         math.exp
func.func @only_tanh(%arg0: vector<8xf32>) -> (vector<8xf32>, vector<8xf32>) {
  %0 = math.tanh %arg0 : vector<8xf32>
  %1 = math.exp %arg0 : vector<8xf32>
  return %0, %1 : vector<8xf32>, vector<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.math.polynomial_approximation ops = ["tanh"]
    } : !transform.any_op
    transform.yield
  }
}

// -----

// rsqrt approximation requires `enable_avx2`, even when listed in `ops`.

// CHECK-LABEL: func @rsqrt_no_avx2
// CHECK:         math.rsqrt
func.func @rsqrt_no_avx2(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.rsqrt %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.math.polynomial_approximation ops = ["rsqrt"]
    } : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @rsqrt_avx2
// CHECK-NOT:     math.rsqrt
// CHECK:         x86.avx.rsqrt
// CHECK-NOT:     math.rsqrt
func.func @rsqrt_avx2(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.rsqrt %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.math.polynomial_approximation ops = ["rsqrt"] enable_avx2
    } : !transform.any_op
    transform.yield
  }
}
