use crate::Value;
use std::collections::HashMap;

pub trait Func<InKind, In, Out> {
    fn apply(&self, input: In) -> Out;
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
    let output = func.apply(input);
    let output_tensors = output.tensors();
    let mut jvps = HashMap::new();
    for t in output_tensors.tensor_iter() {
        jvps.insert(t.id(), t.jvp(&mut grads));
    }
    let grad = OUT::grad(&output_tensors, &jvps);
    (output, grad)
}

pub type BoxedFunc<IN, OUT> = Box<dyn Fn(IN) -> OUT>;

pub fn linearize<K, IN, OUT, F>(func: F, input: IN) -> (OUT, BoxedFunc<IN::Gradient, OUT::Gradient>)
where
    F: Func<K, IN, OUT>,
    IN: Value,
    OUT: Value,
{
    let input_tensors = input.tensors();
    let output = func.apply(input);
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
    (output, Box::new(jvp_fn))
}

pub fn vjp<K, IN, OUT, F>(func: F, input: IN) -> (OUT, BoxedFunc<OUT::Gradient, IN::Gradient>)
where
    F: Func<K, IN, OUT>,
    IN: Value,
    OUT: Value,
{
    let input_tensors = input.tensors();
    let output = func.apply(input);
    let output_tensors = output.tensors();
    let vjps_fn = move |cotangents: OUT::Gradient| {
        let mut cotangent_cache = HashMap::new();
        let mut grads_sum = HashMap::new();
        for i in input_tensors.tensor_iter() {
            grads_sum.insert(i.id(), i.zeros_like());
        }
        OUT::grad_map(&output_tensors, cotangents, &mut cotangent_cache);
        for t in output_tensors.tensor_iter() {
            let cotangent = cotangent_cache
                .get(&t.id())
                .cloned()
                .unwrap_or_else(|| t.ones_like());
            t.vjp(&cotangent, &mut grads_sum);
        }
        let mut vjps = HashMap::new();
        for t in input_tensors.tensor_iter() {
            let id = t.id();
            let c = grads_sum.get(&id).unwrap().clone();
            vjps.insert(id, c);
        }
        IN::grad(&input_tensors, &vjps)
    };
    (output, Box::new(vjps_fn))
}

#[derive(Clone)]
pub struct GradFunc<K, IN, OUT, F>
where
    F: Func<K, IN, OUT> + Clone,
{
    func: F,
    phantom: std::marker::PhantomData<(K, IN, OUT)>,
}

impl<K, IN, OUT, F> GradFunc<K, IN, OUT, F>
where
    F: Func<K, IN, OUT> + Clone,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<K, IN, OUT, F> Func<K, IN, IN::Gradient> for GradFunc<K, IN, OUT, F>
where
    F: Func<K, IN, OUT> + Clone,
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

pub fn grad<K, IN, OUT, F>(func: F) -> GradFunc<K, IN, OUT, F>
where
    F: Func<K, IN, OUT> + Clone,
    IN: Value,
    OUT: Value,
{
    GradFunc::new(func)
}

#[derive(Clone)]
pub struct ValueAndGradFunc<K, IN, OUT, F>
where
    F: Func<K, IN, OUT> + Clone,
{
    func: F,
    phantom: std::marker::PhantomData<(K, IN, OUT)>,
}

impl<K, IN, OUT, F> ValueAndGradFunc<K, IN, OUT, F>
where
    F: Func<K, IN, OUT> + Clone,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<K, IN, OUT, F> Func<K, IN, (OUT, IN::Gradient)> for ValueAndGradFunc<K, IN, OUT, F>
where
    F: Func<K, IN, OUT> + Clone,
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

pub fn value_and_grad<K, IN, OUT, F>(func: F) -> ValueAndGradFunc<K, IN, OUT, F>
where
    F: Func<K, IN, OUT> + Clone,
    IN: Value,
    OUT: Value,
{
    ValueAndGradFunc::new(func)
}
