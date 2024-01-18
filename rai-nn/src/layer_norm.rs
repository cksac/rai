use std::{collections::HashMap, fmt::Debug};

use rai_core::{Backend, DType, Module, ModuleValue, Tensor, TrainableModule, ValueSpec};

use crate::{gather_params, update_params};

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

impl ValueSpec for LayerNorm {
    type Kind = ModuleValue;
    type Tensors = HashMap<usize, Tensor>;
    type Gradient = HashMap<usize, Tensor>;
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
}

impl TrainableModule for LayerNorm {
    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        gather_params!(?self.weight, params);
        gather_params!(?self.bias, params);
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        update_params!(?self.weight, params);
        update_params!(?self.bias, params);
    }
}
