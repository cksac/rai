use crate::{Dim, Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

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

impl Op for ScatterAdd {
    fn clone_boxed(&self) -> Box<dyn Op> {
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

#[track_caller]
pub fn scatter_add(
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
    let inputs = vec![x.clone(), source.clone(), index.clone()];
    Tensor::new(device, dtype, shape, ScatterAdd::new(dim), inputs).into()
}

pub trait ScatterAddOp {
    fn scatter_add<D: Dim>(
        self,
        dim: D,
        index: impl TryAsTensor,
        source: impl TryAsTensor,
    ) -> RaiResult<Tensor>;
}

impl<T> ScatterAddOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn scatter_add<D: Dim>(
        self,
        dim: D,
        index: impl TryAsTensor,
        source: impl TryAsTensor,
    ) -> RaiResult<Tensor> {
        scatter_add(self, dim, index, source)
    }
}
