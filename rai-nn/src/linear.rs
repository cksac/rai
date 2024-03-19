use rai_core::{AsDevice, Shape, Tensor, Type};
use rai_derive::Module;

use crate::init::{self, Init, DEFAULT_KAIMING_NORMAL};

#[derive(Clone, Debug, Module)]
#[module(crate = rai_core)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    #[inline]
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        dtype: impl Type,
        device: impl AsDevice,
    ) -> Self {
        let bound = 1. / (in_features as f64).sqrt();
        let bias_init = match has_bias {
            true => Some(init::Uniform::new(-bound, bound)),
            false => None,
        };
        Self::new_with_init(
            in_features,
            out_features,
            dtype,
            device,
            DEFAULT_KAIMING_NORMAL,
            bias_init,
        )
    }

    pub fn new_with_init(
        in_features: usize,
        out_features: usize,
        dtype: impl Type,
        device: impl AsDevice,
        weight_init: impl Init,
        bias_init: Option<impl Init>,
    ) -> Self {
        let device = device.device();
        let weight = weight_init.new_tensor([out_features, in_features], dtype, device);
        let bias = bias_init.map(|init| init.new_tensor([out_features], dtype, device));
        Self { weight, bias }
    }

    pub fn new_with_params(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
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
}
