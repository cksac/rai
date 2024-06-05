use crate::{Func, GradMap, TensorIter, Value};

pub fn jvp<K, IN, OUT, F>(func: F, input: IN, tangents: IN::Gradient) -> (OUT, OUT::Gradient)
where
    F: Func<K, IN, OUT>,
    IN: Value,
    OUT: Value,
{
    let input_tensors = input.tensors();
    let mut grads = GradMap::with_capacity(input_tensors.count());
    IN::grad_map(&input_tensors, tangents, &mut grads);
    let output = func.invoke(input);
    let output_tensors = output.tensors();
    let mut jvps = GradMap::with_capacity(output_tensors.count());
    for t in output_tensors.tensor_iter() {
        jvps.insert(t.id(), t.jvp(&mut grads));
    }
    let grad = OUT::grad(&output_tensors, &jvps);
    (output, grad)
}
