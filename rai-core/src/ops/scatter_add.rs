use crate::{Dim, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

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
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("ScatterAdd")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("ScatterAdd({})", self.dim)
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        let tangent_source = &tangents[1];
        let index = &primals[2];
        tangent_x.scatter_add(self.dim, index, tangent_source)
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let index = &primals[2];
        let cotangent_x = cotangent.clone();
        let cotangent_source = cotangent.gather(self.dim, index);
        vec![cotangent_x, cotangent_source]
    }
}

#[track_caller]
pub fn scatter_add(x: &Tensor, dim: impl Dim, index: &Tensor, source: &Tensor) -> Tensor {
    let dim = x.dim(dim);
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let inputs = vec![x.clone(), source.clone(), index.clone()];
    Tensor::new(device, dtype, shape, ScatterAdd::new(dim), inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn scatter_add(
        &self,
        dim: impl Dim,
        index: impl AsRef<Tensor>,
        source: impl AsRef<Tensor>,
    ) -> Tensor {
        scatter_add(self, dim, index.as_ref(), source.as_ref())
    }
}
