use crate::{broadcast_binary_op, impl_std_ops, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Div;

impl Op for Div {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Div")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let rhs = &primals[1];
        let tangent_lhs = &tangents[0];
        let tangent_rhs = &tangents[1];
        tangent_lhs / rhs - output * tangent_rhs
    }

    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let rhs = &primals[1];
        let cotangent_lhs = cotangent / rhs;
        let cotangent_rhs = -cotangent * output / rhs;
        vec![cotangent_lhs, cotangent_rhs]
    }
}

broadcast_binary_op!(
    /// Divides two `Tensor` objects.
    ///
    /// Arguments
    ///
    /// * `lhs` - The first `Tensor`.
    /// * `rhs` - The second `Tensor`.
    ///
    /// Returns
    ///
    /// The resulting `Tensor` after the division.
    Div,
    div
);

impl_std_ops!(Div, div);

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn div<T>(&self, rhs: T) -> Tensor
    where
        for<'a> &'a Self: std::ops::Div<T, Output = Tensor>,
    {
        std::ops::Div::div(self, rhs)
    }
}
