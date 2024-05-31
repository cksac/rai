use crate::{Primitive, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgMin {
    pub dim: usize,
    pub keep_dim: bool,
}

impl ArgMin {
    pub fn new(dim: usize, keep_dim: bool) -> Self {
        Self { dim, keep_dim }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Primitive for ArgMin {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("ArgMin({}, {})", &self.dim, &self.keep_dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        output.zeros_like()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        vec![]
    }
}
