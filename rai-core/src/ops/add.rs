use crate::{broadcast_binary_op, impl_std_ops, Op, RaiResult, Shape, Tensor};
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
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> RaiResult<Tensor> {
        let tangent_lhs = &tangents[0];
        let tangent_rhs = &tangents[1];
        tangent_lhs + tangent_rhs
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(
        &self,
        _output: &Tensor,
        _primals: &[Tensor],
        cotangent: &Tensor,
    ) -> RaiResult<Vec<Tensor>> {
        let cotangent_lhs = cotangent.clone();
        let cotangent_rhs = cotangent.clone();
        Ok(vec![cotangent_lhs, cotangent_rhs]).into()
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

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn add<RHS>(&self, rhs: RHS) -> RaiResult<Tensor>
    where
        for<'a> &'a Self: std::ops::Add<RHS, Output = RaiResult<Tensor>>,
    {
        self + rhs
    }
}
