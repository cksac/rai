use rai_core::{Backend, DType, Module, Tensor};
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        dtype: DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Self {
        let backend = backend.into();
        // TODO: init strategy
        let weight = Tensor::normal([out_features, in_features], dtype, backend.clone());
        let bias = if has_bias {
            Some(Tensor::normal([out_features], dtype, backend))
        } else {
            None
        };
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, input: Tensor) -> Tensor {
        match &self.bias {
            Some(bias) => input.matmul(self.weight.t()) + bias,
            None => input.matmul(self.weight.t()),
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        match &self.bias {
            Some(bias) => vec![self.weight.clone(), bias.clone()],
            None => vec![self.weight.clone()],
        }
    }
}
