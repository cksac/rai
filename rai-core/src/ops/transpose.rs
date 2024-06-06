use crate::{Dim, Op, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transpose {
    pub dim0: usize,
    pub dim1: usize,
}

impl Transpose {
    pub fn new(dim0: usize, dim1: usize) -> Self {
        Self { dim0, dim1 }
    }

    pub fn dim0(&self) -> usize {
        self.dim0
    }

    pub fn dim1(&self) -> usize {
        self.dim1
    }
}

impl Op for Transpose {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Transpose({}, {})", self.dim0(), self.dim1())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.transpose(self.dim0, self.dim1)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_x = cotangent.transpose(self.dim0, self.dim1);
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn transpose(x: &Tensor, dim0: impl Dim, dim1: impl Dim) -> Tensor {
    let dim0 = x.dim(dim0);
    let dim1 = x.dim(dim1);
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape_transpose(dim0, dim1);
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Transpose::new(dim0, dim1), inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn t(&self) -> Tensor {
        transpose(self, -2, -1)
    }

    #[inline]
    #[track_caller]
    pub fn transpose(&self, dim0: impl Dim, dim1: impl Dim) -> Tensor {
        transpose(self, dim0, dim1)
    }
}
