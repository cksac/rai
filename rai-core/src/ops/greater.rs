use crate::{broadcast_binary_op, Op, RaiResult, Shape, Tensor, TensorOps, TryAsTensor, U8};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Greater;

impl Op for Greater {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, _primals: &[Tensor], _tangents: &[Tensor]) -> Tensor {
        output.zeros_like()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], _cotangent: &Tensor) -> Vec<Tensor> {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let cotangent_lhs = lhs.zeros_like();
        let cotangent_rhs = rhs.zeros_like();
        vec![cotangent_lhs, cotangent_rhs]
    }
}

broadcast_binary_op!(Greater, gt, U8);

pub trait GtOp {
    fn gt(self, rhs: impl TryAsTensor) -> RaiResult<Tensor>;
}

impl<T> GtOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn gt(self, rhs: impl TryAsTensor) -> RaiResult<Tensor> {
        gt(self, rhs)
    }
}
