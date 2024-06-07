use crate::{Dim, Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

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
    fn clone_boxed(&self) -> Box<dyn Op> {
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

#[track_caller]
pub fn index_select(
    x: impl TryAsTensor,
    dim: impl Dim,
    index: impl TryAsTensor,
) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let index = crate::try_get! { index.try_as_tensor() };
    let dim = x.dim(dim);
    let device = x.device();
    let dtype = x.dtype();
    let mut shape = x.shape().to_vec();
    shape[dim] = index.elem_count();
    let inputs = vec![x.clone(), index.clone()];
    // TODO: asserts
    Tensor::new(device, dtype, shape, IndexSelect::new(dim), inputs).into()
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn index_select(&self, dim: impl Dim, index: impl TryAsTensor) -> RaiResult<Tensor> {
        index_select(self, dim, index)
    }
}
