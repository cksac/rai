use std::collections::HashMap;

use crate::{Shape, Value};

pub trait Func<IN, OUT> {
    fn apply(&self, input: IN) -> OUT;
}

mod tensor_iter;
pub use tensor_iter::TensorIter;

mod eval;
pub use eval::eval;

mod fn_impls;

mod raiexpr;
pub use raiexpr::raiexpr;

pub fn jvp<IN, OUT, F>(func: F, input: IN, tangents: IN::Gradient) -> (OUT, OUT::Gradient)
where
    F: Func<IN, OUT>,
    IN: Value,
    OUT: Value,
{
    let input_tensors = input.tensors();
    let mut grads = HashMap::new();
    IN::grad_map(&input_tensors, tangents, &mut grads);
    let output = func.apply(input);
    let output_tensors = output.tensors();
    let mut jvps = HashMap::new();
    for t in output_tensors.tensor_iter() {
        jvps.insert(t.id(), t.jvp(&mut grads));
    }
    let grad = OUT::grad(&output_tensors, &jvps);
    (output, grad)
}

type VjpFunc<IN, OUT> = Box<dyn Fn(IN) -> OUT>;
pub fn vjp<IN, OUT, F>(func: F, input: IN) -> (OUT, VjpFunc<OUT::Gradient, IN::Gradient>)
where
    F: Func<IN, OUT>,
    IN: Value,
    OUT: Value,
{
    let input_tensors = input.tensors();
    let output = func.apply(input);
    let output_tensors = output.tensors();
    let vjps_fn = move |cotangents: OUT::Gradient| {
        let mut grads = HashMap::new();
        OUT::grad_map(&output_tensors, cotangents, &mut grads);
        for t in output_tensors.tensor_iter() {
            t.vjp(&mut grads);
        }
        let mut vjps = HashMap::new();
        for t in input_tensors.tensor_iter() {
            let id = t.id();
            let c = grads.get(&id).unwrap().clone();
            if c.shape_eq(&t) {
                vjps.insert(id, c);
            } else {
                // reduce sum cotangent to its input shape
                // TODO: will cotangent ndim always > input ndim?
                vjps.insert(id, c.sum(..t.ndim()));
            }
        }
        IN::grad(&input_tensors, &vjps)
    };
    (output, Box::new(vjps_fn))
}

#[derive(Clone)]
pub struct GradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone,
{
    func: F,
    phantom: std::marker::PhantomData<(IN, OUT)>,
}

impl<IN, OUT, F> GradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<IN, OUT, F> Func<IN, IN::Gradient> for GradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone,
    IN: Value,
    OUT: Value,
{
    fn apply(&self, input: IN) -> IN::Gradient {
        let (output, vjp_fn) = vjp(self.func.clone(), input);
        let mut cotangents = HashMap::new();
        let output_tensors = output.tensors();
        for t in output_tensors.tensor_iter() {
            cotangents.insert(t.id(), t.ones_like());
        }
        vjp_fn(OUT::grad(&output_tensors, &cotangents))
    }
}

pub fn grad<IN, OUT, F>(func: F) -> GradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone,
    IN: Value,
    OUT: Value,
{
    GradFunc::new(func)
}

#[derive(Clone)]
pub struct ValueAndGradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone,
{
    func: F,
    phantom: std::marker::PhantomData<(IN, OUT)>,
}

impl<IN, OUT, F> ValueAndGradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<IN, OUT, F> Func<IN, (OUT, IN::Gradient)> for ValueAndGradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone,
    IN: Value,
    OUT: Value,
{
    fn apply(&self, input: IN) -> (OUT, IN::Gradient) {
        let (output, vjp_fn) = vjp(self.func.clone(), input);
        let mut cotangents = HashMap::new();
        let output_tensors = output.tensors();
        for t in output_tensors.tensor_iter() {
            cotangents.insert(t.id(), t.ones_like());
        }
        (output, vjp_fn(OUT::grad(&output_tensors, &cotangents)))
    }
}

pub fn value_and_grad<IN, OUT, F>(func: F) -> ValueAndGradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone,
    IN: Value,
    OUT: Value,
{
    ValueAndGradFunc::new(func)
}
