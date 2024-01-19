use std::{collections::HashMap, fmt::Debug};

use rai_core::{nn::Module, trainable_module, Backend, DType, Shape, Tensor};

use crate::{gather_named_params, gather_params, update_params};

pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn new(
        dims: usize,
        eps: f32,
        dtype: impl DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Self {
        let backend = &backend.into();
        let weight = Tensor::ones([dims], dtype, backend);
        Self { weight, eps }
    }
}

impl Module for RMSNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Self::Output {
        let s = 1.0 / (x.shape_at(-1) as f32).sqrt();
        let n = ((x * s).square().sum((-1, true)) + self.eps).rsqrt();
        &self.weight * x * n
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        gather_params!(params, self.weight);
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        update_params!(params, self.weight);
    }

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        gather_named_params!(params, prefix, "w", self.weight);
    }
}

trainable_module!(RMSNorm);
