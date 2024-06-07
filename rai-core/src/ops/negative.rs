use crate::{Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Negative;

impl Op for Negative {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        -tangent_x
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_x = -cotangent;
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn neg(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Negative, inputs).into()
}

impl std::ops::Neg for Tensor {
    type Output = RaiResult<Tensor>;

    #[track_caller]
    fn neg(self) -> Self::Output {
        neg(self)
    }
}

impl<'a> std::ops::Neg for &'a Tensor {
    type Output = RaiResult<Tensor>;

    #[track_caller]
    fn neg(self) -> Self::Output {
        neg(self)
    }
}

impl std::ops::Neg for RaiResult<Tensor> {
    type Output = RaiResult<Tensor>;

    #[track_caller]
    fn neg(self) -> Self::Output {
        neg(self)
    }
}

impl<'a> std::ops::Neg for &'a RaiResult<Tensor> {
    type Output = RaiResult<Tensor>;

    #[track_caller]
    fn neg(self) -> Self::Output {
        neg(self)
    }
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn neg(&self) -> RaiResult<Tensor> {
        neg(self)
    }
}
