use crate::{ElemType, Tensor};
use std::fmt::Debug;

pub trait ClampBound: Debug {
    fn bound(&self, input: &Tensor) -> Tensor;
}

impl<T: ElemType> ClampBound for T {
    fn bound(&self, input: &Tensor) -> Tensor {
        input.full_like(*self).to_dtype(input)
    }
}

impl ClampBound for Tensor {
    fn bound(&self, input: &Tensor) -> Tensor {
        self.to_dtype(input)
    }
}

impl ClampBound for &Tensor {
    fn bound(&self, input: &Tensor) -> Tensor {
        (*self).to_dtype(input)
    }
}

#[track_caller]
pub fn clamp(x: &Tensor, min: impl ClampBound, max: impl ClampBound) -> Tensor {
    let min = min.bound(x);
    let max = max.bound(x);
    x.maximum(min).minimum(max)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn clamp(&self, min: impl ClampBound, max: impl ClampBound) -> Tensor {
        clamp(self, min, max)
    }
}
