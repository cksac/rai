use crate::{broadcast_binary_op, impl_std_ops, Op, RaiResult, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Div;

impl Op for Div {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let rhs = &primals[1];
        let tangent_lhs = &tangents[0];
        let tangent_rhs = &tangents[1];
        tangent_lhs / rhs - output * tangent_rhs
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
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
    /// # Arguments
    ///
    /// * `lhs` - The first `Tensor`.
    /// * `rhs` - The second `Tensor`.
    ///
    /// # Returns
    ///
    /// The resulting `Tensor` after the division.
    Div,
    div
);

impl_std_ops!(Div, div);

pub trait DivOp {
    #[inline]
    #[track_caller]
    fn sub<RHS>(self, rhs: RHS) -> RaiResult<Tensor>
    where
        Self: Sized + std::ops::Div<RHS, Output = RaiResult<Tensor>>,
    {
        self / rhs
    }
}

impl DivOp for Tensor {}
impl<'a> DivOp for &'a Tensor {}
impl DivOp for RaiResult<Tensor> {}
impl<'a> DivOp for RaiResult<&'a Tensor> {}
impl<'a> DivOp for &'a RaiResult<Tensor> {}
