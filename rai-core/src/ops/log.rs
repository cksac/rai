use crate::{Op, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Log;

impl Op for Log {
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
        tangent_x / x
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent / x;
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn log(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Log, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn log(&self) -> Tensor {
        log(self)
    }
}
