use crate::{broadcast_binary_op, impl_std_ops, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mul;

impl Op for Mul {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Mul")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let tangent_lhs = &tangents[0];
        let tangent_rhs = &tangents[1];
        tangent_lhs * rhs + tangent_rhs * lhs
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let cotangent_lhs = cotangent * rhs;
        let cotangent_rhs = cotangent * lhs;
        vec![cotangent_lhs, cotangent_rhs]
    }
}

broadcast_binary_op!(
    /// Multiplies two `Tensor` objects.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The first `Tensor`.
    /// * `rhs` - The second `Tensor`.
    ///
    /// # Returns
    ///
    /// The resulting `Tensor` after the multiplication.
    Mul,
    mul
);

impl_std_ops!(Mul, mul);

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn mul<T>(&self, rhs: T) -> Tensor
    where
        for<'a> &'a Self: std::ops::Mul<T, Output = Tensor>,
    {
        std::ops::Mul::mul(self, rhs)
    }
}
