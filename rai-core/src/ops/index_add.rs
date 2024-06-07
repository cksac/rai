use crate::{Dim, Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

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
    fn clone_boxed(&self) -> Box<dyn Op> {
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

#[track_caller]
pub fn index_add(
    x: impl TryAsTensor,
    dim: impl Dim,
    index: impl TryAsTensor,
    source: impl TryAsTensor,
) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let index = crate::try_get! { index.try_as_tensor() };
    let source = crate::try_get! { source.try_as_tensor() };
    let dim = x.dim(dim);
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    // due to vjp only return by position
    // x and source will have grads, therefore it comes first
    let inputs = vec![x.clone(), source.clone(), index.clone()];
    // TODO: asserts
    Tensor::new(device, dtype, shape, IndexAdd::new(dim), inputs).into()
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn index_add(
        &self,
        dim: impl Dim,
        index: impl TryAsTensor,
        source: impl TryAsTensor,
    ) -> RaiResult<Tensor> {
        index_add(self, dim, index, source)
    }
}
