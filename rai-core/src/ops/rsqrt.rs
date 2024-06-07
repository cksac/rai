use crate::{Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Rsqrt;

impl Op for Rsqrt {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        -0.5 * tangent_x * (x.rsqrt() / x)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = -0.5 * cotangent * (x.rsqrt() / x);
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn rsqrt(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Rsqrt, inputs).into()
}

pub trait RsqrtOp {
    fn rsqrt(self) -> RaiResult<Tensor>;
}

impl<T> RsqrtOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn rsqrt(self) -> RaiResult<Tensor> {
        rsqrt(self)
    }
}
