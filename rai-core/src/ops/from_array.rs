use crate::{Op, Tensor, Type};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct FromArray<D>
where
    D: Type,
{
    pub data: Vec<D::Repr>,
}

impl<D> FromArray<D>
where
    D: Type,
{
    pub fn new(data: impl Into<Vec<D::Repr>>) -> Self {
        Self { data: data.into() }
    }
}

impl<D> Op for FromArray<D>
where
    D: Type,
{
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        "FromArray(...)".to_string()
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
