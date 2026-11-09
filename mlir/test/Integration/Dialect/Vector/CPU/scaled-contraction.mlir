// RUN: mlir-opt %s -transform-interpreter -test-transform-dialect-erase-schedule \
// RUN:   -lower-vector-multi-reduction \
// RUN:   -arith-expand="include-f4e2m1=true include-f8e4m3fn=true include-f8e5m2=true include-f8e8m0=true" \
// RUN:   -test-lower-to-llvm | \
// RUN: mlir-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

// Scaled contractions are decomposed into standard vector operations which
// are then lowered to LLVM.

#block_scaling_trait = {
  indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                   affine_map<(m, n, k) -> (m, k floordiv 32)>,
                   affine_map<(m, n, k) -> (n, k)>,
                   affine_map<(m, n, k) -> (n, k floordiv 32)>,
                   affine_map<(m, n, k) -> (m, n)>],
  iterator_types = ["parallel", "parallel", "reduction"]
}

#block16_scaling_trait = {
  indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                   affine_map<(m, n, k) -> (m, k floordiv 16)>,
                   affine_map<(m, n, k) -> (n, k)>,
                   affine_map<(m, n, k) -> (n, k floordiv 16)>,
                   affine_map<(m, n, k) -> (m, n)>],
  iterator_types = ["parallel", "parallel", "reduction"]
}

#dim_scaling_trait = {
  indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                   affine_map<(m, n, k) -> (m)>,
                   affine_map<(m, n, k) -> (n, k)>,
                   affine_map<(m, n, k) -> (n)>,
                   affine_map<(m, n, k) -> (m, n)>],
  iterator_types = ["parallel", "parallel", "reduction"]
}

#element_scaling_trait = {
  indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                   affine_map<(m, n, k) -> (m, k)>,
                   affine_map<(m, n, k) -> (n, k)>,
                   affine_map<(m, n, k) -> ()>,
                   affine_map<(m, n, k) -> (m, n)>],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// MXFP8 with mixed input formats and a NaN scale.
func.func @mxfp8() {
  %a = arith.constant dense<1.0> : vector<2x64xf8E5M2>
  %b = arith.constant dense<2.0> : vector<2x64xf8E4M3FN>
  %sa = arith.constant dense<[[1.0, 2.0], [4.0, 0.5]]> : vector<2x2xf8E8M0FNU>
  %sb = arith.constant dense<[[1.0, 1.0], [2.0, 0.25]]> : vector<2x2xf8E8M0FNU>
  %c = arith.constant dense<0.0> : vector<2x2xf32>
  %0 = vector.scaled_contract #block_scaling_trait %a, %sa, %b, %sb, %c
    : vector<2x64xf8E5M2>, vector<2x2xf8E8M0FNU>,
      vector<2x64xf8E4M3FN>, vector<2x2xf8E8M0FNU> into vector<2x2xf32>
  vector.print %0 : vector<2x2xf32>

  %sa_nan = arith.constant dense<[[1.0, 0xFF], [4.0, 0.5]]> : vector<2x2xf8E8M0FNU>
  %1 = vector.scaled_contract #block_scaling_trait %a, %sa_nan, %b, %sb, %c
    : vector<2x64xf8E5M2>, vector<2x2xf8E8M0FNU>,
      vector<2x64xf8E4M3FN>, vector<2x2xf8E8M0FNU> into vector<2x2xf32>
  vector.print %1 : vector<2x2xf32>
  return
}

// MXFP4 lhs with E8M0 scales and FP8 rhs with E4M3 scales.
func.func @fp4_fp8() {
  %a = arith.constant dense<1.5> : vector<2x32xf4E2M1FN>
  %b = arith.constant dense<-2.0> : vector<2x32xf8E4M3FN>
  %sa = arith.constant dense<[[2.0, 0.5], [1.0, 4.0]]> : vector<2x2xf8E8M0FNU>
  %sb = arith.constant dense<[[1.5, 0.75], [1.0, 1.0]]> : vector<2x2xf8E4M3FN>
  %c = arith.constant dense<1.0> : vector<2x2xf32>
  %0 = vector.scaled_contract #block16_scaling_trait %a, %sa, %b, %sb, %c
    : vector<2x32xf4E2M1FN>, vector<2x2xf8E8M0FNU>,
      vector<2x32xf8E4M3FN>, vector<2x2xf8E4M3FN> into vector<2x2xf32>
  vector.print %0 : vector<2x2xf32>
  return
}

// Per-row and per-column scaling.
func.func @dim_scaling() {
  %a = arith.constant dense<1.0> : vector<2x64xf32>
  %b = arith.constant dense<1.0> : vector<2x64xf32>
  %sa = arith.constant dense<[1.0, 2.0]> : vector<2xf32>
  %sb = arith.constant dense<[3.0, 0.5]> : vector<2xf32>
  %c = arith.constant dense<1.0> : vector<2x2xf32>
  %0 = vector.scaled_contract #dim_scaling_trait %a, %sa, %b, %sb, %c
    : vector<2x64xf32>, vector<2xf32>,
      vector<2x64xf32>, vector<2xf32> into vector<2x2xf32>
  vector.print %0 : vector<2x2xf32>
  return
}

// Scaling of every element along the reduction dim with f16 output.
func.func @element_scaling() {
  %a = arith.constant dense<2.0> : vector<2x4xf8E4M3FN>
  %b = arith.constant dense<[[1.0, 1.0, 1.0, 1.0], [-1.0, 0.5, 0.5, 0.0]]> : vector<2x4xf8E4M3FN>
  %sa = arith.constant dense<[[1.0, 2.0, 4.0, 8.0], [0.5, 0.5, 0.5, 0.5]]> : vector<2x4xf8E8M0FNU>
  %sb = arith.constant 0.5 : f16
  %c = arith.constant dense<0.5> : vector<2x2xf16>
  %0 = vector.scaled_contract #element_scaling_trait %a, %sa, %b, %sb, %c
    : vector<2x4xf8E4M3FN>, vector<2x4xf8E8M0FNU>,
      vector<2x4xf8E4M3FN>, f16 into vector<2x2xf16>
  vector.print %0 : vector<2x2xf16>
  return
}

func.func @entry() {
  // D[i, j] = 64 * (sa[i, 0] * sb[j, 0] + sa[i, 1] * sb[j, 1])
  // CHECK:      ( ( 192, 160 ), ( 288, 520 ) )
  // CHECK-NEXT: ( ( nan, nan ), ( 288, 520 ) )
  call @mxfp8() : () -> ()

  // D[i, j] = -48 * (sa[i, 0] * sb[j, 0] + sa[i, 1] * sb[j, 1]) + 1
  // CHECK-NEXT: ( ( -161, -119 ), ( -215, -239 ) )
  call @fp4_fp8() : () -> ()

  // D[i, j] = 64 * sa[i] * sb[j] + 1
  // CHECK-NEXT: ( ( 193, 33 ), ( 385, 65 ) )
  call @dim_scaling() : () -> ()

  // D[i, j] = SUM_k (2 * sa[i, k]) * (b[j, k] * 0.5) + 0.5
  // CHECK-NEXT: ( ( 15.5, 2.5 ), ( 2.5, 0.5 ) )
  call @element_scaling() : () -> ()
  return
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
