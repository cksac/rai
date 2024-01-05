use rai_core::{Module, Tensor};

#[derive(Clone, Debug, Copy)]
pub struct Relu;

impl Module for Relu {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }
}
