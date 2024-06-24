use crate::{Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Rsqrt;

impl Op for Rsqrt {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Rsqrt")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        -0.5 * tangent_x * (x.rsqrt() / x)
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = -0.5 * cotangent * (x.rsqrt() / x);
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn rsqrt(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Rsqrt, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn rsqrt(&self) -> Tensor {
        rsqrt(self)
    }
}
