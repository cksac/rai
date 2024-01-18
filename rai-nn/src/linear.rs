use std::{collections::HashMap, fmt::Debug};

use rai_core::{trainable_module, Backend, DType, Module, Tensor};

use crate::{gather_params, update_params};

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
        dtype: impl DType,
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
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Self::Output {
        match &self.bias {
            Some(bias) => x.matmul(self.weight.t()) + bias,
            None => x.matmul(self.weight.t()),
        }
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        gather_params!(self.weight, params);
        gather_params!(?self.bias, params);
    }

    #[track_caller]
    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        update_params!(self.weight, params);
        update_params!(?self.bias, params);
    }
}

trainable_module!(Linear);
