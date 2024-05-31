use crate::{Primitive, Tensor};
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

impl Primitive for Transpose {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
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
