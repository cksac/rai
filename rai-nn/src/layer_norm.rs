use std::{collections::HashMap, fmt::Debug};

use rai_core::{nn::Module, trainable_module, Backend, DType, Tensor};

use crate::{gather_params, update_params, NamedParameter};

pub struct LayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(
        dims: usize,
        eps: f32,
        affine: bool,
        dtype: impl DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Self {
        let backend = &backend.into();
        let (weight, bias) = if affine {
            let weight = Some(Tensor::ones([dims], dtype, backend));
            let bias = Some(Tensor::zeros([dims], dtype, backend));
            (weight, bias)
        } else {
            (None, None)
        };
        Self { weight, bias, eps }
    }
}

impl Module for LayerNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Self::Output {
        let mean = x.mean((-1, true));
        let var = x.var((-1, true));
        let x = (x - mean) * (var + self.eps).rsqrt();
        if let Some(weight) = &self.weight {
            let bias = self.bias.as_ref().unwrap();
            weight * x * bias
        } else {
            x
        }
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        gather_params!(params, ?self.weight);
        gather_params!(params, ?self.bias);
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        update_params!(params, ?self.weight);
        update_params!(params, ?self.bias);
    }

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        self.weight.gather_to(params, prefix, "w");
        self.bias.gather_to(params, prefix, "b");
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        self.weight.update_by(params, prefix, "w");
        self.bias.update_by(params, prefix, "b");
    }
}

trainable_module!(LayerNorm);
