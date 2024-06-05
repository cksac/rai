use crate::{vjp, Func, GradMap, TensorIter, Value};

pub fn grad<'a, K, IN, OUT, F>(func: F) -> impl Fn(IN) -> IN::Gradient + Clone + 'a
where
    F: Func<K, IN, OUT> + Clone + 'a,
    IN: Value,
    OUT: Value,
{
    move |input: IN| {
        let (output, vjp_fn) = vjp(func.clone(), input);
        let output_tensors = output.tensors();
        let mut grads = GradMap::with_capacity(output_tensors.count());
        for t in output_tensors.tensor_iter() {
            grads.insert(t.id(), t.ones_like());
        }
        vjp_fn(OUT::grad(&output_tensors, &grads))
    }
}
