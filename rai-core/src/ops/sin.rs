use crate::{Op, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Sin;
impl Op for Sin {
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
        x.cos() * tangent_x
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent * x.cos();
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn sin(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Sin, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn sin(&self) -> Tensor {
        sin(self)
    }
}
