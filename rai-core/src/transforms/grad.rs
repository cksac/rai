use crate::{vjp, Func, TensorIter, Value};
use std::collections::HashMap;

pub fn grad<'a, K, IN, OUT, F>(func: F) -> impl Fn(IN) -> IN::Gradient + Clone + 'a
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
        vjp_fn(OUT::grad(&output_tensors, &cotangents))
    }
}
