use crate::{utils::topological_sort_filter, Func, GradMap, TensorIter, Value};

pub fn vjp<'a, K, IN, OUT, F>(
    func: F,
    input: IN,
) -> (OUT, impl Fn(OUT::Gradient) -> IN::Gradient + Clone + 'a)
where
    F: Func<K, IN, OUT> + Clone + 'a,
    IN: Value,
    OUT: Value,
{
    let input_tensors = input.tensors();
    let output = func.invoke(input);
    let output_tensors = output.tensors();
    let vjps_fn = move |cotangents: OUT::Gradient| {
        let mut grads = GradMap::with_capacity(output_tensors.count());
        OUT::grad_map(&output_tensors, cotangents, &mut grads);
        let tape = topological_sort_filter(&output_tensors, |t| !t.inputs().is_empty());
        // run the tape backwards
        for t in tape.iter().rev() {
            let primals = &*t.inputs();
            let cotangent = grads.entry(t.id()).or_insert_with(|| t.ones_like());
            let cotangents = t.primitive().vjp(t, primals, cotangent);
            for (primal, cotan) in primals.iter().zip(cotangents.into_iter()) {
                let id = primal.id();
                if let Some(sum) = grads.get(id) {
                    grads.insert(id, sum + cotan);
                } else {
                    grads.insert(id, cotan);
                }
            }
        }
        IN::grad(&input_tensors, &grads)
    };
    (output, vjps_fn)
}
