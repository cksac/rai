use crate::{Dim, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

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

impl Op for IndexAdd {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("IndexAdd")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("IndexAdd({:?})", &self.dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        let tangent_source = &tangents[1];
        let index = &primals[2];
        tangent_x.index_add(self.dim, index, tangent_source)
    }

    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let index = &primals[2];
        let cotangent_x = cotangent.clone();
        let cotangent_source = cotangent.index_select(self.dim, index);
        vec![cotangent_x, cotangent_source]
    }
}

#[track_caller]
pub fn index_add(x: &Tensor, dim: impl Dim, index: &Tensor, source: &Tensor) -> Tensor {
    let dim = x.dim(dim);
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    // due to vjp only return by position
    // x and source will have grads, therefore it comes first
    let inputs = vec![x.clone(), source.clone(), index.clone()];
    // TODO: asserts
    Tensor::new(device, dtype, shape, IndexAdd::new(dim), inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn index_add(&self, dim: impl Dim, index: &Tensor, source: &Tensor) -> Tensor {
        index_add(self, dim, index, source)
    }
}
