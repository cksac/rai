use crate::{Op, Shape, Tensor};
use std::any::Any;
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
