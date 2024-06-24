use crate::{broadcast_binary_op, Op, Shape, Tensor, U8};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Less;

impl Op for Less {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Less")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, output: &Tensor, _primals: &[Tensor], _tangents: &[Tensor]) -> Tensor {
        output.zeros_like()
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], _cotangent: &Tensor) -> Vec<Tensor> {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let cotangent_lhs = lhs.zeros_like();
        let cotangent_rhs = rhs.zeros_like();
        vec![cotangent_lhs, cotangent_rhs]
    }
}

broadcast_binary_op!(Less, lt, U8);

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn lt<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        lt(self, rhs.as_ref())
    }
}
