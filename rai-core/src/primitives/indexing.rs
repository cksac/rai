use crate::{Primitive, Shape, Tensor};
use std::any::Any;
use tracing::Level;

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
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        let index = &primals[1];
        tangent_x.gather(self.dim, index)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let index = &primals[1];
        let source = x.zeros_like();
        let cotangent_x = source.scatter_add(self.dim, index, cotangent);
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexAdd {
    pub dim: usize,
}

impl IndexAdd {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn dim(&self) -> &usize {
        &self.dim
    }
}

impl Primitive for IndexAdd {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("IndexAdd({:?})", &self.dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        let tangent_source = &tangents[1];
        let index = &primals[2];
        tangent_x.index_add(self.dim, index, tangent_source)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let index = &primals[2];
        let cotangent_x = cotangent.clone();
        let cotangent_source = cotangent.index_select(self.dim, index);
        vec![cotangent_x, cotangent_source]
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
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        let index = &primals[1];
        tangent_x.index_select(self.dim, index)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let index = &primals[1];
        let source = x.zeros_like();
        let cotangent_x = source.index_add(self.dim, index, cotangent);
        vec![cotangent_x]
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
        let tangent_x = &tangents[0];
        tangent_x.narrow(self.dim, self.start, self.len)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let x_dim = x.shape();
        let left_pad = if self.start == 0 {
            None
        } else {
            let mut dims = x_dim.to_vec();
            dims[self.dim] = self.start;

            Some(Tensor::zeros(dims, cotangent.dtype(), cotangent.device()))
        };
        let right_pad = x_dim[self.dim] - self.start - self.len;
        let right_pad = if right_pad == 0 {
            None
        } else {
            let mut dims = x_dim.to_vec();
            dims[self.dim] = right_pad;
            Some(Tensor::zeros(dims, cotangent.dtype(), cotangent.device()))
        };
        let cotangent_x = match (left_pad, right_pad) {
            (None, None) => cotangent.clone(),
            (Some(l), None) => Tensor::cat(&[&l, cotangent], self.dim),
            (None, Some(r)) => Tensor::cat(&[cotangent, &r], self.dim),
            (Some(l), Some(r)) => Tensor::cat(&[&l, cotangent, &r], self.dim),
        };
        vec![cotangent_x]
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
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let pred = &primals[2];
        let tangent_t = &tangents[0];
        let tangent_f = &tangents[1];
        pred.where_cond(tangent_t, tangent_f)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let pred = &primals[2];
        let zeros = &cotangent.zeros_like();
        let contangent_t = pred.where_cond(cotangent, zeros);
        let contangent_f = pred.where_cond(zeros, cotangent);
        vec![contangent_t, contangent_f]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScatterAdd {
    pub dim: usize,
}

impl ScatterAdd {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Primitive for ScatterAdd {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("ScatterAdd({})", self.dim)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        let tangent_source = &tangents[1];
        let index = &primals[2];
        tangent_x.scatter_add(self.dim, index, tangent_source)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let index = &primals[2];
        let cotangent_x = cotangent.clone();
        let cotangent_source = cotangent.gather(self.dim, index);
        vec![cotangent_x, cotangent_source]
    }
}
