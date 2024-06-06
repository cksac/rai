use crate::{Op, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Sign;

impl Op for Sign {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], _tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        x.zeros_like()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], _cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = x.zeros_like();
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn sign(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Sign, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn sign(&self) -> Tensor {
        sign(self)
    }
}
