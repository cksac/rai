use crate::{Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Tanh;

impl Op for Tanh {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        (tangent_x + tangent_x * output) * (x.ones_like() - output)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = (cotangent + cotangent * output) * (x.ones_like() - output);
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn tanh(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Tanh, inputs).into()
}

pub trait TanhOp {
    fn tanh(self) -> RaiResult<Tensor>;
}

impl<T> TanhOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn tanh(self) -> RaiResult<Tensor> {
        tanh(self)
    }
}
