use crate::{Primitive, Tensor, Type};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Full<D>
where
    D: Type,
{
    pub val: D::Repr,
}
impl<D> Full<D>
where
    D: Type,
{
    pub fn new(val: D::Repr) -> Self {
        Full { val }
    }
}

impl<D> Primitive for Full<D>
where
    D: Type,
{
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(*self)
    }

    fn dot_label(&self) -> String {
        format!("Full({:?})", self.val)
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
