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

    fn captured_inputs(&self) -> Option<Vec<Tensor>> {
        Some(self.parameters())
    }
}
