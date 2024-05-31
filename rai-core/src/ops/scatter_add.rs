use crate::{Op, Tensor};
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
