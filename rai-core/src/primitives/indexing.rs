use std::any::Any;

use tracing::Level;

use crate::{Primitive, Tensor};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gather {
    pub dim: usize,
}

impl Gather {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn dim(&self) -> &usize {
        &self.dim
    }
}

impl Primitive for Gather {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("Gather({:?})", &self.dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        // let tangent_x = &tangents[0];
        // tangent_x.sum((self.dims(), false))
        todo!()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        // let x = &primals[0];
        // let mut shape = x.shape().to_vec();
        // for dim in self.dims() {
        //     shape[*dim] = 1;
        // }
        // let cotangent_x = cotangent.reshape(&shape).broadcast_to(x);
        // vec![cotangent_x]
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexSelect {
    pub dim: usize,
}

impl IndexSelect {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn dim(&self) -> &usize {
        &self.dim
    }
}

impl Primitive for IndexSelect {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("IndexSelect({:?})", &self.dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        // let tangent_x = &tangents[0];
        // tangent_x.sum((self.dims(), false))
        todo!()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        // let x = &primals[0];
        // let mut shape = x.shape().to_vec();
        // for dim in self.dims() {
        //     shape[*dim] = 1;
        // }
        // let cotangent_x = cotangent.reshape(&shape).broadcast_to(x);
        // vec![cotangent_x]
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Narrow {
    pub dim: usize,
    pub start: usize,
    pub len: usize,
}

impl Narrow {
    pub fn new(dim: usize, start: usize, len: usize) -> Self {
        Self { dim, start, len }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn start(&self) -> usize {
        self.start
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Primitive for Narrow {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Narrow({}, {}, {})", self.dim, self.start, self.len)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Where;

impl Primitive for Where {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        todo!()
    }
}
