use crate::{broadcast_binary_op, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Maximum;

impl Op for Maximum {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Maximum")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let tangent_lhs = &tangents[0];
        let tangent_rhs = &tangents[1];
        let lhs_mask = &output.eq(lhs).to_dtype(tangent_lhs);
        let rhs_mask = &output.eq(rhs).to_dtype(tangent_rhs);
        tangent_lhs * lhs_mask / (rhs_mask + 1.0) + tangent_rhs * rhs_mask / (lhs_mask + 1.0)
    }

    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let lhs_mask = &output.eq(lhs).to_dtype(cotangent);
        let rhs_mask = &output.eq(rhs).to_dtype(cotangent);
        let cotangent_lhs = cotangent * lhs_mask / (rhs_mask + 1.0);
        let cotangent_rhs = cotangent * rhs_mask / (lhs_mask + 1.0);
        vec![cotangent_lhs, cotangent_rhs]
    }
}

broadcast_binary_op!(Maximum, maximum);

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn maximum<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        maximum(self, rhs.as_ref())
    }
}
