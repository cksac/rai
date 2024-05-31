use crate::{Primitive, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Add;

impl Primitive for Add {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
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
