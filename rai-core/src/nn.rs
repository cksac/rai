use std::collections::HashMap;

use crate::{Differentiable, Tensor};

pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>);

    fn params(&self) -> HashMap<usize, Tensor> {
        let mut out = HashMap::new();
        self.gather_params(&mut out);
        out
    }

    // TODO: params should be a reference? for model with shared parameters in different layers
    fn update_params(&self, params: &mut HashMap<usize, Tensor>);
}

pub trait DifferentiableModule:
    Module + Differentiable<Tensors = HashMap<usize, Tensor>, Gradient = HashMap<usize, Tensor>>
{
}

impl<'a, T> Module for &'a T
where
    T: Module,
{
    fn forward(&self, x: &Tensor) -> Tensor {
        (*self).forward(x)
    }

    fn gather_params(&self, out: &mut HashMap<usize, Tensor>) {
        (*self).gather_params(out)
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        (*self).update_params(params)
    }
}
