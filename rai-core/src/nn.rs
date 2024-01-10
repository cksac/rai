use crate::{transforms::Func, Tensor, WithTensors};
use std::collections::BTreeMap;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn update(&self, _params: &mut BTreeMap<usize, Tensor>) {}
}

impl<T> Func<Tensor, Tensor> for T
where
    T: Module,
{
    type Tangent = BTreeMap<usize, Tensor>;
    type Cotangent = BTreeMap<usize, Tensor>;

    fn call(&self, input: Tensor) -> Tensor {
        self.forward(&input)
    }

    fn capture_inputs(&self, _input: &Tensor) -> Vec<Tensor> {
        self.parameters()
    }
}

pub struct Aux<T>(pub T);

impl<T> WithTensors for (Tensor, Aux<T>) {
    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone()]
    }
}

// for loss fn (module, input) -> loss
impl<'m, 'i, M, F> Func<(&'m M, &'i Tensor), Tensor> for F
where
    M: Module,
    F: Fn(&'m M, &'i Tensor) -> Tensor,
{
    type Tangent = BTreeMap<usize, Tensor>;
    type Cotangent = BTreeMap<usize, Tensor>;
    fn call(&self, input: (&'m M, &'i Tensor)) -> Tensor {
        self(input.0, input.1)
    }

    fn capture_inputs(&self, input: &(&'m M, &'i Tensor)) -> Vec<Tensor> {
        input.0.parameters()
    }
}

// for loss fn (module, input, label) -> loss
impl<'m, 'i, 'l, M, F> Func<(&'m M, &'i Tensor, &'l Tensor), Tensor> for F
where
    M: Module,
    F: Fn(&'m M, &'i Tensor, &'l Tensor) -> Tensor,
{
    type Tangent = BTreeMap<usize, Tensor>;
    type Cotangent = BTreeMap<usize, Tensor>;
    fn call(&self, input: (&'m M, &'i Tensor, &'l Tensor)) -> Tensor {
        self(input.0, input.1, input.2)
    }

    fn capture_inputs(&self, input: &(&'m M, &'i Tensor, &'l Tensor)) -> Vec<Tensor> {
        input.0.parameters()
    }
}

// for loss fn  (module, input) -> (loss, Aux<T>)
impl<'m, 'i, M, F, T> Func<(&'m M, &'i Tensor), (Tensor, Aux<T>)> for F
where
    M: Module,
    F: Fn(&'m M, &'i Tensor) -> (Tensor, Aux<T>),
{
    type Tangent = BTreeMap<usize, Tensor>;
    type Cotangent = BTreeMap<usize, Tensor>;
    fn call(&self, input: (&'m M, &'i Tensor)) -> (Tensor, Aux<T>) {
        self(input.0, input.1)
    }

    fn capture_inputs(&self, input: &(&'m M, &'i Tensor)) -> Vec<Tensor> {
        input.0.parameters()
    }
}

// for loss fn (module, input, label) -> (loss, Aux<T>)
impl<'m, 'i, 'l, M, F, T> Func<(&'m M, &'i Tensor, &'l Tensor), (Tensor, Aux<T>)> for F
where
    M: Module,
    F: Fn(&'m M, &'i Tensor, &'l Tensor) -> (Tensor, Aux<T>),
{
    type Tangent = BTreeMap<usize, Tensor>;
    type Cotangent = BTreeMap<usize, Tensor>;
    fn call(&self, input: (&'m M, &'i Tensor, &'l Tensor)) -> (Tensor, Aux<T>) {
        self(input.0, input.1, input.2)
    }

    fn capture_inputs(&self, input: &(&'m M, &'i Tensor, &'l Tensor)) -> Vec<Tensor> {
        input.0.parameters()
    }
}
