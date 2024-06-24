use crate::{Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Negative;

impl Op for Negative {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Negative")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        -tangent_x
    }

    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_x = -cotangent;
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn neg(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Negative, inputs)
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;

    #[track_caller]
    fn neg(self) -> Self::Output {
        neg(&self)
    }
}

impl<'a> std::ops::Neg for &'a Tensor {
    type Output = Tensor;

    #[track_caller]
    fn neg(self) -> Self::Output {
        neg(self)
    }
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn neg(&self) -> Tensor {
        neg(self)
    }
}
