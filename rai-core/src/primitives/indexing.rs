use std::any::Any;

use tracing::Level;

use crate::{Primitive, Shape, Tensor};

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

impl Primitive for Gather {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("Gather({:?})", &self.dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        // let tangent_x = &tangents[0];
        // tangent_x.sum((self.dims(), false))
        todo!()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        // let x = &primals[0];
        // let mut shape = x.shape().to_vec();
        // for dim in self.dims() {
        //     shape[*dim] = 1;
        // }
        // let cotangent_x = cotangent.reshape(&shape).broadcast_to(x);
        // vec![cotangent_x]
        todo!()
    }
}
