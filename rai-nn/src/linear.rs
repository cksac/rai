use rai_core::{Backend, DType, Module, Tensor};
use std::{collections::HashMap, fmt::Debug};

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
        let backend = &backend.into();
        // TODO: init strategy
        let weight = Tensor::normal([out_features, in_features], dtype, backend);
        let bias = if has_bias {
            Some(Tensor::normal([out_features], dtype, backend))
        } else {
            None
        };
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        match &self.bias {
            Some(bias) => input.matmul(self.weight.t()) + bias,
            None => input.matmul(self.weight.t()),
        }
    }

    fn gather_parameters(&self, out: &mut Vec<Tensor>) {
        out.push(self.weight.clone());
        if let Some(bias) = &self.bias {
            out.push(bias.clone());
        }
    }

    #[track_caller]
    fn update(&self, params: &mut HashMap<usize, Tensor>) {
        if let Some(weight) = params.remove(&self.weight.id()) {
            self.weight.replace_data(weight);
        }
        if let Some(bias) = &self.bias {
            if let Some(new_bias) = params.remove(&bias.id()) {
                bias.replace_data(new_bias);
            }
        }
    }
}
