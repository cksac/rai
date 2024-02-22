use rai_core::{AsDevice, Shape, Tensor, Type};
use rai_derive::Module;

#[derive(Clone, Debug, Module)]
#[module(crate = rai_core)]
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

    pub fn new_with(weight: Tensor, bias: Option<Tensor>) -> Self {
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
