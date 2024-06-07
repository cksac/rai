use crate::{Dim, Op, RaiResult, Shape, Tensor, TryAsTensor};
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

impl Op for Gather {
    fn clone_boxed(&self) -> Box<dyn Op> {
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

#[track_caller]
pub fn gather(x: impl TryAsTensor, dim: impl Dim, index: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let index = crate::try_get! { index.try_as_tensor() };
    let dim = x.dim(dim);
    assert_eq!(x.rank(), index.rank());
    let mut lhs_shape = x.shape().to_vec();
    lhs_shape.remove(dim);
    let mut idx_shape = index.shape().to_vec();
    idx_shape.remove(dim);
    assert_eq!(lhs_shape, idx_shape);
    let device = x.device();
    let dtype = x.dtype();
    let shape = index.shape();
    let inputs = vec![x.clone(), index.clone()];
    Tensor::new(device, dtype, shape, Gather::new(dim), inputs).into()
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn gather(&self, dim: impl Dim, index: impl TryAsTensor) -> RaiResult<Tensor> {
        gather(self, dim, index)
    }
}
