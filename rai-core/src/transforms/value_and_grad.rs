use crate::{vjp, Func, TensorIter, Value};
use std::collections::HashMap;

pub fn value_and_grad<'a, K, IN, OUT, F>(func: F) -> impl Fn(IN) -> (OUT, IN::Gradient) + Clone + 'a
where
    F: Func<K, IN, OUT> + Clone + 'a,
    IN: Value,
    OUT: Value,
{
    move |input: IN| {
        let (output, vjp_fn) = vjp(func.clone(), input);
        let mut cotangents = HashMap::new();
        let output_tensors = output.tensors();
        for t in output_tensors.tensor_iter() {
            cotangents.insert(t.id(), t.ones_like());
        }
        (output, vjp_fn(OUT::grad(&output_tensors, &cotangents)))
    }
}
