use crate::{ElemType, RaiResult, Tensor, TryAsTensor};
use std::fmt::Debug;

pub trait ClampBound: Debug {
    fn bound(&self, input: impl TryAsTensor) -> RaiResult<Tensor>;
}

impl<T: ElemType> ClampBound for T {
    fn bound(&self, input: impl TryAsTensor) -> RaiResult<Tensor> {
        let input = crate::try_get! { input.try_as_tensor() };
        input.full_like(*self).to_dtype(input)
    }
}

impl ClampBound for Tensor {
    fn bound(&self, input: impl TryAsTensor) -> RaiResult<Tensor> {
        let input = crate::try_get! { input.try_as_tensor() };
        self.to_dtype(input)
    }
}

impl<'a> ClampBound for &'a Tensor {
    fn bound(&self, input: impl TryAsTensor) -> RaiResult<Tensor> {
        let input = crate::try_get! { input.try_as_tensor() };
        (*self).to_dtype(input)
    }
}

impl ClampBound for RaiResult<Tensor> {
    fn bound(&self, input: impl TryAsTensor) -> RaiResult<Tensor> {
        let input = crate::try_get! { input.try_as_tensor() };
        self.to_dtype(input)
    }
}

impl<'a> ClampBound for &'a RaiResult<Tensor> {
    fn bound(&self, input: impl TryAsTensor) -> RaiResult<Tensor> {
        let input = crate::try_get! { input.try_as_tensor() };
        (*self).to_dtype(input)
    }
}

#[track_caller]
pub fn clamp(x: impl TryAsTensor, min: impl ClampBound, max: impl ClampBound) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let min = min.bound(x);
    let max = max.bound(x);
    x.maximum(min).minimum(max)
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn clamp(&self, min: impl ClampBound, max: impl ClampBound) ->  RaiResult<Tensor> {
        clamp(self, min, max)
    }
}
