use crate::{Primitive, Tensor, Type};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct Arange<D: Type> {
    pub start: D::Repr,
    pub stop: D::Repr,
    pub step: D::Repr,
}

impl<D: Type> Arange<D> {
    pub fn new(start: D::Repr, stop: D::Repr, step: D::Repr) -> Self {
        Self { start, stop, step }
    }
}

impl<D: Type> Primitive for Arange<D> {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("Arange({:?}, {:?}, {:?})", self.start, self.stop, self.step)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    #[inline]
    fn jvp(&self, output: &Tensor, _primals: &[Tensor], _tangents: &[Tensor]) -> Tensor {
        output.ones_like()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    #[inline]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], _cotangent: &Tensor) -> Vec<Tensor> {
        vec![]
    }
}
