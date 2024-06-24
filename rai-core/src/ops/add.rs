use crate::{broadcast_binary_op, impl_std_ops, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Add;

impl Op for Add {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Add")
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
        tangent_lhs + tangent_rhs
    }

    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_lhs = cotangent.clone();
        let cotangent_rhs = cotangent.clone();
        vec![cotangent_lhs, cotangent_rhs]
    }
}

broadcast_binary_op!(
    /// Adds two `Tensor` objects.
    ///
    /// Arguments
    ///
    /// * `lhs` - The first `Tensor`.
    /// * `rhs` - The second `Tensor`.
    ///
    /// Returns
    ///
    /// The resulting `Tensor` after the addition.
    Add,
    add
);

impl_std_ops!(Add, add);

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn add<T>(&self, rhs: T) -> Tensor
    where
        for<'a> &'a Self: std::ops::Add<T, Output = Tensor>,
    {
        std::ops::Add::add(self, rhs)
    }
}
