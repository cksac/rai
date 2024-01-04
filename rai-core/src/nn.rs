use std::collections::BTreeMap;

use crate::{transforms::Func, Tensor};

pub trait Module {
    fn forward(&self, input: Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
}

impl<T> Func<Tensor, Tensor> for T
where
    T: Module,
{
    type Tangent = BTreeMap<usize, Tensor>;
    type Cotangent = BTreeMap<usize, Tensor>;

    fn call(&self, input: Tensor) -> Tensor {
        self.forward(input)
    }

    fn capture_inputs(&self, input: &Tensor) -> Vec<Tensor> {
        self.parameters()
    }
}

// for loss fn return loss
impl<M, F> Func<(&M, Tensor), Tensor> for F
where
    M: Module,
    F: Fn(&M, &Tensor) -> Tensor,
{
    type Tangent = BTreeMap<usize, Tensor>;
    type Cotangent = BTreeMap<usize, Tensor>;
    fn call(&self, input: (&M, Tensor)) -> Tensor {
        self(input.0, &input.1)
    }

    fn capture_inputs(&self, input: &(&M, Tensor)) -> Vec<Tensor> {
        input.0.parameters()
    }
}

// for loss fn return (loss, logits)
impl<M, F> Func<(&M, Tensor), (Tensor, Tensor)> for F
where
    M: Module,
    F: Fn(&M, &Tensor) -> (Tensor, Tensor),
{
    type Tangent = BTreeMap<usize, Tensor>;
    type Cotangent = BTreeMap<usize, Tensor>;
    fn call(&self, input: (&M, Tensor)) -> (Tensor, Tensor) {
        self(input.0, &input.1)
    }

    fn capture_inputs(&self, input: &(&M, Tensor)) -> Vec<Tensor> {
        input.0.parameters()
    }
}
