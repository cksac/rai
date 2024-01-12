use std::collections::HashMap;

use rai_core::{non_differentiable, Module, Tensor};

#[derive(Clone, Debug, Copy)]
pub struct Relu;

impl Module for Relu {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }
}

non_differentiable!(Relu);
