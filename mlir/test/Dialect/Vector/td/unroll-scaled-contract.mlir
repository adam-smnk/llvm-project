module attributes {transform.with_named_sequence} {
  transform.named_sequence @unroll_scaled_contract(%module_op: !transform.any_op {transform.readonly}) {

    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.vector.unroll_scaled_contract [2, 4, 16]
    } : !transform.any_op

    transform.yield
  }
}
