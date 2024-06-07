use crate::{Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::{any::Any, f64::consts::PI};
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Erf;

impl Op for Erf {
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
        (2. / PI.sqrt()) * (x.square().neg()).exp() * tangent_x
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = (2. / PI.sqrt()) * (x.square().neg()).exp() * cotangent;
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn erf(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Erf, inputs).into()
}

pub trait ErfOp {
    fn erf(self) -> RaiResult<Tensor>;
}

impl<T> ErfOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn erf(self) -> RaiResult<Tensor> {
        erf(self)
    }
}
