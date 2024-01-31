use rai_core::{AsDevice, Shape, Tensor, Type};
use rai_derive::Module;

#[derive(Debug, Clone, Module)]
#[module(crate = rai_core)]
pub struct RMSNorm {
    weight: Tensor,
    #[param(skip)]
    eps: f32,
}

impl RMSNorm {
    pub fn new(dims: usize, eps: f32, dtype: impl Type, device: impl AsDevice) -> Self {
        let weight = Tensor::ones([dims], dtype, device);
        Self { weight, eps }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let s = 1.0 / (x.shape_at(-1) as f32).sqrt();
        let n = ((x * s).square().sum((-1, true)) + self.eps).rsqrt();
        &self.weight * x * n
    }
}
