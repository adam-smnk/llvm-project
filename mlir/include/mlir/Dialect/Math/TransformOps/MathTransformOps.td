//===- MathTransformOps.td - Math transform ops ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MATH_TRANSFORM_OPS
#define MATH_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"

def ApplyPolynomialApproximationPatternsOp : Op<Transform_Dialect,
    "apply_patterns.math.polynomial_approximation",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collect patterns rewriting math functions (e.g. `exp`, `log`, `tanh`,
    `erf`, `sin`) as polynomial approximations in `arith` and `math.fma` ops,
    on scalars and vectors. Targets without vector math library calls (e.g.
    LLVM lowering a vector `math.exp` to one scalar `expf` call per element)
    then keep the computation vectorized. Operations on types narrower than
    f32 are computed in f32.

    `ops` restricts the patterns to a subset of the operations, given by
    mnemonic (e.g. `ops = ["exp", "tanh"]`). If it is empty, the default set
    of `populateMathPolynomialApproximationPatterns` is used.

    `enable_avx2` additionally enables approximations that emit AVX2
    intrinsics through the X86 dialect (currently `rsqrt`). Without it, these
    operations are left unchanged even if they are listed in `ops`.
  }];

  let arguments = (ins DefaultValuedAttr<StrArrayAttr, "{}">:$ops,
                       UnitAttr:$enable_avx2);
  let assemblyFormat =
      "(`ops` `=` $ops^)? (`enable_avx2` $enable_avx2^)? attr-dict";
}

#endif // MATH_TRANSFORM_OPS
