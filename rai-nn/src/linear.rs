use crate::{gather_params, update_params, NamedParameter};
use rai_core::{nn::Module, trainable_module, AsDevice, Shape, Tensor, Type};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new<T: Type>(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        dtype: T,
        device: impl AsDevice,
    ) -> Self {
        let device = device.device();
        // TODO: init strategy
        let weight = Tensor::rand([out_features, in_features], dtype, device);
        let bias = if has_bias {
            Some(Tensor::rand([out_features], dtype, device))
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
        // todo: move the broadcast checking to matmul?
        let w = &match x.shape() {
            [b1, b2, _, _] => self.weight.broadcast_left([*b1, *b2]).t(),
            [b, _, _] => self.weight.broadcast_left([*b]).t(),
            _ => self.weight.t(),
        };
        match &self.bias {
            Some(bias) => x.matmul(w) + bias,
            None => x.matmul(w),
        }
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        gather_params!(params, self.weight);
        gather_params!(params, ?self.bias);
    }

    #[track_caller]
    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        update_params!(params, self.weight);
        update_params!(params, ?self.bias);
    }

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        self.weight.gather_to(params, prefix, "weight");
        self.bias.gather_to(params, prefix, "bias");
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        self.weight.update_by(params, prefix, "weight");
        self.bias.update_by(params, prefix, "bias");
    }
}

trainable_module!(Linear);
