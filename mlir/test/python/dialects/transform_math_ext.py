# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import math


def run_apply_patterns(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            sequence = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                [],
                transform.AnyOpType.get(),
            )
            with InsertionPoint(sequence.body):
                apply = transform.ApplyPatternsOp(sequence.bodyTarget)
                with InsertionPoint(apply.patterns):
                    f()
                transform.YieldOp()
        print("\nTEST:", f.__name__)
        print(module)
    return f


@run_apply_patterns
def testPolynomialApproximationDefault():
    math.ApplyPolynomialApproximationPatternsOp()
    # CHECK-LABEL: TEST: testPolynomialApproximationDefault
    # CHECK: apply_patterns
    # CHECK: transform.apply_patterns.math.polynomial_approximation{{$}}


@run_apply_patterns
def testPolynomialApproximationOps():
    math.ApplyPolynomialApproximationPatternsOp(ops=["exp", "tanh"], enable_avx2=True)
    # CHECK-LABEL: TEST: testPolynomialApproximationOps
    # CHECK: apply_patterns
    # CHECK: transform.apply_patterns.math.polynomial_approximation
    # CHECK-SAME: ops = ["exp", "tanh"] enable_avx2
