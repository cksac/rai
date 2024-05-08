use crate::Value;
use std::collections::{BTreeSet, HashMap};

pub trait Func<InKind, In, Out> {
    fn invoke(&self, input: In) -> Out;
}

mod tensor_iter;
pub use tensor_iter::TensorIter;

mod eval;
pub use eval::eval;

mod fn_impls;

mod raiexpr;
pub use raiexpr::raiexpr;

pub fn jvp<K, IN, OUT, F>(func: F, input: IN, tangents: IN::Gradient) -> (OUT, OUT::Gradient)
where
    F: Func<K, IN, OUT>,
    IN: Value,
    OUT: Value,
{
    let input_tensors = input.tensors();
    let mut grads = HashMap::new();
    IN::grad_map(&input_tensors, tangents, &mut grads);
    let output = func.invoke(input);
    let output_tensors = output.tensors();
    let mut jvps = HashMap::new();
    for t in output_tensors.tensor_iter() {
        jvps.insert(t.id(), t.jvp(&mut grads));
    }
    let grad = OUT::grad(&output_tensors, &jvps);
    (output, grad)
}

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
        let mut grads = HashMap::new();
        IN::grad_map(&input_tensors, tangents, &mut grads);
        let mut jvps = HashMap::new();
        for t in output_tensors.tensor_iter() {
            jvps.insert(t.id(), t.jvp(&mut grads));
        }
        OUT::grad(&output_tensors, &jvps)
    };
    (output, jvp_fn)
}

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
        let mut tape = BTreeSet::new();
        let mut stack = Vec::new();
        // use iterative instead of recursive to avoid stack overflow
        // TODO: use proper topo sort algorithm, now sort by id in BTreeSet
        for output in output_tensors.tensor_iter() {
            stack.push(output.clone());
        }
        while let Some(t) = stack.pop() {
            if tape.contains(&t) || t.inputs().is_empty() {
                continue;
            }
            tape.insert(t.clone());
            for input in t.inputs().iter() {
                stack.push(input.clone());
            }
        }
        // run the tape backwards
        for t in tape.iter().rev() {
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
