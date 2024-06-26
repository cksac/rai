use crate::{Func, GradMap, TensorIter, Value};

pub fn linearize<'a, K, IN, OUT, F>(
    func: F,
    input: IN,
) -> (OUT, impl Fn(IN::Gradient) -> OUT::Gradient + Clone + 'a)
where
    F: Func<K, IN, OUT> + Clone + 'a,
    IN: Value,
    OUT: Value,
{
    let input_tensors = input.tensors();
    let output = func.invoke(input);
    let output_tensors = output.tensors();
    let jvp_fn = move |tangents: IN::Gradient| {
        let mut grads = GradMap::with_capacity(input_tensors.count());
        IN::grad_map(&input_tensors, tangents, &mut grads);
        let mut jvps = GradMap::with_capacity(output_tensors.count());
        for t in output_tensors.tensor_iter() {
            jvps.insert(t.id(), t.jvp(&mut grads));
        }
        OUT::grad(&output_tensors, &jvps)
    };
    (output, jvp_fn)
}
