use crate::{try_get, Error, Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::{any::Any, rc::Rc};
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Abs;

impl Op for Abs {
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
        tangent_x * x.sign()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent * x.sign();
        vec![cotangent_x]
    }
}

pub fn abs(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = try_get! { x.try_as_tensor() };
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Abs, inputs).into()
}

pub trait AbsOp {
    fn abs(self) -> RaiResult<Tensor>;
}

impl<T> AbsOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn abs(self) -> RaiResult<Tensor> {
        abs(self)
    }
}
