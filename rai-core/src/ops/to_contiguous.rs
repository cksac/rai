use crate::{Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToContiguous;

impl Op for ToContiguous {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("ToContiguous")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.clone()
    }

    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_x = cotangent.clone();
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn to_contiguous(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, ToContiguous, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn to_contiguous(&self) -> Tensor {
        to_contiguous(self)
    }
}
