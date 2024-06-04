use crate::{Func, TensorIter, Value};
use std::collections::{HashMap, HashSet};

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
        let mut cotangent_cache = HashMap::new();
        OUT::grad_map(&output_tensors, cotangents, &mut cotangent_cache);
        let mut visited = HashSet::new();
        let mut tape = Vec::new();
        let mut stack = Vec::new();
        for output in output_tensors.tensor_iter() {
            stack.push(output.clone());
        }
        while let Some(t) = stack.pop() {
            if visited.contains(&t.id()) || t.inputs().is_empty() {
                continue;
            }
            visited.insert(t.id());
            tape.push(t.clone());
            for input in t.inputs().iter() {
                stack.push(input.clone());
            }
        }
        // run the tape backwards
        for t in tape.iter() {
            let primals = &*t.inputs();
            let cotangent = cotangent_cache
                .entry(t.id())
                .or_insert_with(|| t.ones_like());
            let cotangents = t.primitive().vjp(t, primals, cotangent);
            for (primal, cotan) in primals.iter().zip(cotangents.into_iter()) {
                let id = primal.id();
                if let Some(sum) = cotangent_cache.get(&id) {
                    cotangent_cache.insert(id, sum + cotan);
                } else {
                    cotangent_cache.insert(id, cotan);
                }
            }
        }
        // collect the final cotangents for inputs
        let mut vjps = HashMap::new();
        for t in input_tensors.tensor_iter() {
            let id = t.id();
            let c = cotangent_cache.get(&id).unwrap().clone();
            vjps.insert(id, c);
        }
        IN::grad(&input_tensors, &vjps)
    };
    (output, vjps_fn)
}
