use crate::{Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Tanh;

impl Op for Tanh {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Tanh")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        (tangent_x + tangent_x * output) * (x.ones_like() - output)
    }

    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = (cotangent + cotangent * output) * (x.ones_like() - output);
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn tanh(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Tanh, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn tanh(&self) -> Tensor {
        tanh(self)
    }
}
