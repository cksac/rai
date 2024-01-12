use std::collections::HashMap;

use crate::{Differentiable, Tensor};

#[allow(unused_variables)]
pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;

    fn gather_parameters(&self, params: &mut HashMap<usize, Tensor>) {}

    fn parameters(&self) -> HashMap<usize, Tensor> {
        let mut out = HashMap::new();
        self.gather_parameters(&mut out);
        out
    }

    fn update(&self, params: &mut HashMap<usize, Tensor>) {}
}

pub trait DifferentiableModule:
    Module + Differentiable<Tensors = HashMap<usize, Tensor>, Gradient = HashMap<usize, Tensor>>
{
}

impl<'a, T> Module for &'a T
where
    T: Module,
{
    fn forward(&self, input: &Tensor) -> Tensor {
        (*self).forward(input)
    }

    fn gather_parameters(&self, out: &mut HashMap<usize, Tensor>) {
        (*self).gather_parameters(out)
    }

    fn update(&self, params: &mut HashMap<usize, Tensor>) {
        (*self).update(params)
    }
}
