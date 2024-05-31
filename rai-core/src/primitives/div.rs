use crate::{Primitive, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Div;

impl Primitive for Div {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
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
