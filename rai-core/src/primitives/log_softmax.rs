use crate::{Primitive, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogSoftmax {
    pub dim: usize,
}

impl LogSoftmax {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Primitive for LogSoftmax {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("LogSoftmax({:?})", &self.dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x - tangent_x.sum((self.dim, true)) * output.exp()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_x = cotangent - cotangent.sum((self.dim, true)) * output.exp();
        vec![cotangent_x]
    }
}
