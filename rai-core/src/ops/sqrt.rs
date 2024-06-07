use crate::{Op, RaiResult, Shape, Tensor, TryAsTensor};
use safetensors::slice::TensorIndexer;
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Sqrt;

impl Op for Sqrt {
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
        0.5 * tangent_x / x.sqrt()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = 0.5 * cotangent / x.sqrt();
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn sqrt(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Sqrt, inputs).into()
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn sqrt(&self) -> RaiResult<Tensor> {
        sqrt(self)
    }
}
