use crate::{broadcast_binary_op, impl_std_ops, Op, RaiResult, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sub;

impl Op for Sub {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_lhs = &tangents[0];
        let tangent_rhs = &tangents[1];
        tangent_lhs - tangent_rhs
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
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

pub trait SubOp {
    #[inline]
    #[track_caller]
    fn sub<RHS>(self, rhs: RHS) -> RaiResult<Tensor>
    where
        Self: Sized + std::ops::Sub<RHS, Output = RaiResult<Tensor>>,
    {
        self - rhs
    }
}

impl SubOp for Tensor {}
impl<'a> SubOp for &'a Tensor {}
impl SubOp for RaiResult<Tensor> {}
impl<'a> SubOp for RaiResult<&'a Tensor> {}
impl<'a> SubOp for &'a RaiResult<Tensor> {}
