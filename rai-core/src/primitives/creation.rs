use std::any::Any;

use tracing::Level;

use crate::{DType, ElemType, Primitive, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Full<D>
where
    D: DType,
{
    pub val: D::Repr,
}
impl<D> Full<D>
where
    D: DType,
{
    pub fn new(val: D::Repr) -> Self {
        Full { val }
    }
}

impl<D> Primitive for Full<D>
where
    D: DType,
{
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Normal;

impl Primitive for Normal {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(*self)
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

#[derive(Clone, Debug, PartialEq)]
pub struct Arange {
    pub start: f64,
    pub stop: f64,
    pub step: f64,
}

impl Arange {
    pub fn new(start: f64, stop: f64, step: f64) -> Self {
        Self { start, stop, step }
    }
}

impl Primitive for Arange {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("Arange({}, {}, {})", self.start, self.stop, self.step)
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

#[derive(Clone, Debug, PartialEq)]
pub struct FromArray<D>
where
    D: DType,
{
    pub data: Vec<D::Repr>,
}

impl<D> FromArray<D>
where
    D: DType,
{
    pub fn new(data: impl Into<Vec<D::Repr>>) -> Self {
        Self { data: data.into() }
    }
}

impl<D> Primitive for FromArray<D>
where
    D: DType,
{
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("FromArray({:?})", self.data)
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
