use crate::Tensor;
use std::f32::consts::PI;

#[track_caller]
pub fn relu(x: &Tensor) -> Tensor {
    x.maximum(x.zeros_like())
}

#[track_caller]
pub fn relu2(x: &Tensor) -> Tensor {
    x.maximum(x.zeros_like()).sqrt()
}

#[track_caller]
pub fn relu6(x: &Tensor) -> Tensor {
    x.clamp(0.0f32, 6.0f32)
}

#[track_caller]
pub fn gelu(x: &Tensor) -> Tensor {
    x * 0.5f32 * (1.0f32 + (x / 2.0f32.sqrt()).erf())
}

#[track_caller]
pub fn new_gelu(x: &Tensor) -> Tensor {
    0.5f32 * x * (1.0f32 + ((2.0f32 / PI).sqrt() * (x + 0.044715f32 * x.powf(3.0))).tanh())
}

#[track_caller]
pub fn silu(x: &Tensor) -> Tensor {
    x / (x.neg().exp() + 1.0f32)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn relu(&self) -> Tensor {
        relu(self)
    }

    #[inline]
    #[track_caller]
    pub fn relu2(&self) -> Tensor {
        relu2(self)
    }

    #[inline]
    #[track_caller]
    pub fn relu6(&self) -> Tensor {
        relu6(self)
    }

    #[inline]
    #[track_caller]
    pub fn gelu(&self) -> Tensor {
        gelu(self)
    }

    #[inline]
    #[track_caller]
    pub fn new_gelu(&self) -> Tensor {
        new_gelu(self)
    }

    #[inline]
    #[track_caller]
    pub fn silu(&self) -> Tensor {
        silu(self)
    }
}
