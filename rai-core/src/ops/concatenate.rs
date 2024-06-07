use crate::{Dim, Op, RaiResult, Shape, Tensor};
use std::{any::Any, fmt::Debug};
use tracing::Level;

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

impl Op for Concatenate {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Concatenate({:?})", self.dim())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> RaiResult<Tensor> {
        Tensor::cat(tangents, self.dim)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(
        &self,
        _output: &Tensor,
        primals: &[Tensor],
        cotangent: &Tensor,
    ) -> RaiResult<Vec<Tensor>> {
        let mut start_idx = 0;
        let mut cotangent_primals = Vec::with_capacity(primals.len());
        for t in primals {
            let len = t.size(self.dim);
            let cotangent_t = cotangent.narrow(self.dim, start_idx, len);
            cotangent_primals.push(cotangent_t);
            start_idx += len;
        }
        cotangent_primals.into_iter().collect()
    }
}

#[track_caller]
pub fn cat<T: AsRef<Tensor> + Debug>(tensors: &[T], dim: impl Dim) -> RaiResult<Tensor> {
    let inputs: Vec<Tensor> = tensors.iter().map(AsRef::as_ref).cloned().collect();
    let t1 = &inputs[0].clone();
    let dim = t1.dim(dim);
    let device = t1.device();
    let dtype = t1.dtype();
    let mut shape = t1.shape().to_vec();
    shape[dim] = 0;
    for t in inputs.iter() {
        // todo: check shape
        shape[dim] += t.size(dim);
    }
    Tensor::new(device, dtype, shape, Concatenate::new(dim), inputs).into()
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn cat<T: AsRef<Tensor> + Debug>(tensors: &[T], dim: impl Dim) -> RaiResult<Tensor> {
        cat(tensors, dim)
    }
}
