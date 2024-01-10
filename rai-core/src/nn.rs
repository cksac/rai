use crate::{transforms::Func, Tensor, WithTensors};
use std::collections::HashMap;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn gather_parameters(&self, out: &mut Vec<Tensor>);
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        self.gather_parameters(&mut params);
        params
    }
    fn update(&self, _params: &mut HashMap<usize, Tensor>) {}
}

impl<T> Func<Tensor, Tensor> for T
where
    T: Module,
{
    type Tangent = HashMap<usize, Tensor>;
    type Cotangent = HashMap<usize, Tensor>;

    fn call(&self, input: Tensor) -> Tensor {
        self.forward(&input)
    }

    fn self_captured_tensors(&self, tensors: &mut Vec<Tensor>) {
        self.gather_parameters(tensors)
    }

    fn extract_input_tensors(&self, _input: &Tensor, _tensors: &mut Vec<Tensor>) {}
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
    type Tangent = HashMap<usize, Tensor>;
    type Cotangent = HashMap<usize, Tensor>;
    fn call(&self, input: (&'m M, &'i Tensor)) -> Tensor {
        self(input.0, input.1)
    }

    fn extract_input_tensors(&self, input: &(&'m M, &'i Tensor), inputs: &mut Vec<Tensor>) {
        inputs.extend(input.0.parameters())
    }
}

// for loss fn (module, input, label) -> loss
impl<'m, 'i, 'l, M, F> Func<(&'m M, &'i Tensor, &'l Tensor), Tensor> for F
where
    M: Module,
    F: Fn(&'m M, &'i Tensor, &'l Tensor) -> Tensor,
{
    type Tangent = HashMap<usize, Tensor>;
    type Cotangent = HashMap<usize, Tensor>;
    fn call(&self, input: (&'m M, &'i Tensor, &'l Tensor)) -> Tensor {
        self(input.0, input.1, input.2)
    }

    fn extract_input_tensors(
        &self,
        input: &(&'m M, &'i Tensor, &'l Tensor),
        inputs: &mut Vec<Tensor>,
    ) {
        inputs.extend(input.0.parameters())
    }
}

// for loss fn  (module, input) -> (loss, Aux<T>)
impl<'m, 'i, M, F, T> Func<(&'m M, &'i Tensor), (Tensor, Aux<T>)> for F
where
    M: Module,
    F: Fn(&'m M, &'i Tensor) -> (Tensor, Aux<T>),
{
    type Tangent = HashMap<usize, Tensor>;
    type Cotangent = HashMap<usize, Tensor>;
    fn call(&self, input: (&'m M, &'i Tensor)) -> (Tensor, Aux<T>) {
        self(input.0, input.1)
    }

    fn extract_input_tensors(&self, input: &(&'m M, &'i Tensor), inputs: &mut Vec<Tensor>) {
        inputs.extend(input.0.parameters())
    }
}

// for loss fn (module, input, label) -> (loss, Aux<T>)
impl<'m, 'i, 'l, M, F, T> Func<(&'m M, &'i Tensor, &'l Tensor), (Tensor, Aux<T>)> for F
where
    M: Module,
    F: Fn(&'m M, &'i Tensor, &'l Tensor) -> (Tensor, Aux<T>),
{
    type Tangent = HashMap<usize, Tensor>;
    type Cotangent = HashMap<usize, Tensor>;
    fn call(&self, input: (&'m M, &'i Tensor, &'l Tensor)) -> (Tensor, Aux<T>) {
        self(input.0, input.1, input.2)
    }

    fn extract_input_tensors(
        &self,
        input: &(&'m M, &'i Tensor, &'l Tensor),
        inputs: &mut Vec<Tensor>,
    ) {
        inputs.extend(input.0.parameters())
    }
}
