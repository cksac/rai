use rai_core::{AsDevice, Tensor, Type};
use rai_derive::Module;

#[derive(Clone, Debug, Module)]
#[module(crate = rai_core)]
pub struct LayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    #[param(skip)]
    eps: f64,
}

impl LayerNorm {
    pub fn new(
        dims: usize,
        eps: f64,
        affine: bool,
        dtype: impl Type,
        device: impl AsDevice,
    ) -> Self {
        let device = device.device();
        let (weight, bias) = if affine {
            let weight = Some(Tensor::ones([dims], dtype, device));
            let bias = Some(Tensor::zeros([dims], dtype, device));
            (weight, bias)
        } else {
            (None, None)
        };
        Self { weight, bias, eps }
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        let mean = x.mean((-1, true));
        let var = x.var((-1, true));
        let x = (x - mean) * (var + self.eps).rsqrt();
        if let Some(weight) = &self.weight {
            let bias = self.bias.as_ref().unwrap();
            weight * x + bias
        } else {
            x
        }
    }
}
