use crate::{broadcast_binary_op, impl_std_ops, Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Add;

impl Op for Add {
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
        tangent_lhs + tangent_rhs
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_lhs = cotangent.clone();
        let cotangent_rhs = cotangent.clone();
        vec![cotangent_lhs, cotangent_rhs]
    }
}

broadcast_binary_op!(
    /// Adds two `Tensor` objects.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The first `Tensor`.
    /// * `rhs` - The second `Tensor`.
    ///
    /// # Returns
    ///
    /// The resulting `Tensor` after the addition.
    Add,
    add
);

impl_std_ops!(Add, add);

pub trait AddOp {
    #[inline]
    #[track_caller]
    fn add<RHS>(self, rhs: RHS) -> RaiResult<Tensor>
    where
        Self: Sized + std::ops::Add<RHS, Output = RaiResult<Tensor>>,
    {
        self + rhs
    }
}

impl AddOp for Tensor {}
impl<'a> AddOp for &'a Tensor {}
impl AddOp for RaiResult<Tensor> {}
impl<'a> AddOp for RaiResult<&'a Tensor> {}
impl<'a> AddOp for &'a RaiResult<Tensor> {}
