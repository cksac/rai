use std::collections::HashMap;

use crate::{Differentiable, Tensor};

pub trait Module {
    type Input<'i>;
    type Output<'o>;

    fn forward<'i, 'o>(&self, x: Self::Input<'i>) -> Self::Output<'o>;

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>);

    fn params(&self) -> HashMap<usize, Tensor> {
        let mut params = HashMap::new();
        self.gather_params(&mut params);
        params
    }

    // TODO: params should be a reference? for model with shared parameters in different layers
    fn update_params(&self, params: &mut HashMap<usize, Tensor>);
}

pub trait DifferentiableModule:
    Module + Differentiable<Tensors = HashMap<usize, Tensor>, Gradient = HashMap<usize, Tensor>>
{
}

pub trait SimpleModule<'i, 'o>:
    DifferentiableModule<Input<'i> = &'i Tensor, Output<'o> = Tensor>
{
}

impl<'a, T> Module for &'a T
where
    T: Module,
{
    type Input<'i> = T::Input<'i>;
    type Output<'o> = T::Output<'o>;

    fn forward<'i, 'o>(&self, x: Self::Input<'i>) -> Self::Output<'o> {
        (*self).forward(x)
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        (*self).gather_params(params)
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        (*self).update_params(params)
    }
}
