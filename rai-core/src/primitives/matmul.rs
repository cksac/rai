use crate::{Primitive, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatMul;

impl Primitive for MatMul {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        tangents[0].matmul(&primals[1]) + primals[0].matmul(&tangents[1])
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let cotangent_lhs = cotangent.matmul(rhs.t());
        let cotangent_rhs = lhs.t().matmul(cotangent);
        vec![cotangent_lhs, cotangent_rhs]
    }
}
