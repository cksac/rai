use crate::{RaiResult, Tensor, TryAsTensor};
use std::f32::consts::PI;

#[track_caller]
pub fn relu(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    x.maximum(x.zeros_like())
}

#[track_caller]
pub fn relu2(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    x.maximum(x.zeros_like()).sqrt()
}

#[track_caller]
pub fn relu6(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    x.clamp(0.0f32, 6.0f32)
}

#[track_caller]
pub fn gelu(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    x * 0.5f32 * (1.0f32 + (x / 2.0f32.sqrt()).erf())
}

#[track_caller]
pub fn new_gelu(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    0.5f32 * x * (1.0f32 + ((2.0f32 / PI).sqrt() * (x + 0.044715f32 * x.powf(3.0))).tanh())
}

#[track_caller]
pub fn silu(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    x / (x.neg().exp() + 1.0f32)
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn relu(&self) -> RaiResult<Tensor> {
        relu(self)
    }

    #[inline]
    #[track_caller]
    pub fn relu2(&self) -> RaiResult<Tensor> {
        relu2(self)
    }

    #[inline]
    #[track_caller]
    pub fn relu6(&self) -> RaiResult<Tensor> {
        relu6(self)
    }

    #[inline]
    #[track_caller]
    pub fn gelu(&self) -> RaiResult<Tensor> {
        gelu(self)
    }

    #[inline]
    #[track_caller]
    pub fn new_gelu(&self) -> RaiResult<Tensor> {
        new_gelu(self)
    }

    #[inline]
    #[track_caller]
    pub fn silu(&self) -> RaiResult<Tensor> {
        silu(self)
    }
}
