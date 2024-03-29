use crate::{Primitive, Shape, Tensor, Type};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Normal<D: Type> {
    pub mean: D::Repr,
    pub std: D::Repr,
}

impl<D: Type> Normal<D> {
    pub fn new(mean: D::Repr, std: D::Repr) -> Self {
        Self { mean, std }
    }
}

impl<D: Type> Primitive for Normal<D> {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Normal({:?}, {:?})", self.mean, self.std)
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
pub struct Random<D: Type> {
    pub from: D::Repr,
    pub to: D::Repr,
}

impl<D: Type> Random<D> {
    pub fn new(from: D::Repr, to: D::Repr) -> Self {
        Self { from, to }
    }
}

impl<D: Type> Primitive for Random<D> {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Random({:?}, {:?})", self.from, self.to)
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

impl<D> Primitive for FromArray<D>
where
    D: Type,
{
    fn clone_boxed(&self) -> Box<dyn Primitive> {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Concatenate {
    pub dim: usize,
}

impl Concatenate {
    pub fn new(dim: impl Into<usize>) -> Self {
        Self { dim: dim.into() }
    }

    pub fn dim(&self) -> &usize {
        &self.dim
    }
}

impl Primitive for Concatenate {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Concatenate({:?})", self.dim())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        Tensor::cat(tangents, self.dim)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let mut start_idx = 0;
        let mut cotangent_primals = Vec::with_capacity(primals.len());
        for t in primals {
            let len = t.shape_at(self.dim);
            let cotangent_t = cotangent.narrow(self.dim, start_idx, len);
            cotangent_primals.push(cotangent_t);
            start_idx += len;
        }
        cotangent_primals
    }
}
