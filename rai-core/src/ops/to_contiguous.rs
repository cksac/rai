use crate::{Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToContiguous;

impl Op for ToContiguous {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.clone()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_x = cotangent.clone();
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn to_contiguous(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, ToContiguous, inputs).into()
}

pub trait ToContiguousOp {
    fn to_contiguous(self) -> RaiResult<Tensor>;
}

impl<T> ToContiguousOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn to_contiguous(self) -> RaiResult<Tensor> {
        to_contiguous(self)
    }
}
