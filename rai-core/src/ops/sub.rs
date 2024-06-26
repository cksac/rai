use crate::{broadcast_binary_op, impl_std_ops, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sub;

impl Op for Sub {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Sub")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_lhs = &tangents[0];
        let tangent_rhs = &tangents[1];
        tangent_lhs - tangent_rhs
    }

    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_lhs = cotangent.clone();
        let cotangent_rhs = -cotangent;
        vec![cotangent_lhs, cotangent_rhs]
    }
}

broadcast_binary_op!(
    /// Subtracts two `Tensor` objects.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The first `Tensor`.
    /// * `rhs` - The second `Tensor`.
    ///
    /// # Returns
    ///
    /// The resulting `Tensor` after the subtraction.
    Sub,
    sub
);

impl_std_ops!(Sub, sub);

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn sub<T>(&self, rhs: T) -> Tensor
    where
        for<'a> &'a Self: std::ops::Sub<T, Output = Tensor>,
    {
        std::ops::Sub::sub(self, rhs)
    }
}
