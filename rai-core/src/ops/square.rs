use crate::{Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Square;
impl Op for Square {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Square")
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
        2 * x * tangent_x
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = 2 * cotangent * x;
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn square(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Square, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn square(&self) -> Tensor {
        square(self)
    }
}
