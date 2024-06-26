use crate::{Dim, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

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

impl Op for IndexSelect {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("IndexSelect")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("IndexSelect({:?})", &self.dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        let index = &primals[1];
        tangent_x.index_select(self.dim, index)
    }

    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let index = &primals[1];
        let source = x.zeros_like();
        let cotangent_x = source.index_add(self.dim, index, cotangent);
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn index_select(x: &Tensor, dim: impl Dim, index: &Tensor) -> Tensor {
    let dim = x.dim(dim);
    let device = x.device();
    let dtype = x.dtype();
    let mut shape = x.shape().to_vec();
    shape[dim] = index.elem_count();
    let inputs = vec![x.clone(), index.clone()];
    // TODO: asserts
    Tensor::new(device, dtype, shape, IndexSelect::new(dim), inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn index_select(&self, dim: impl Dim, index: impl AsRef<Tensor>) -> Tensor {
        index_select(self, dim, index.as_ref())
    }
}
