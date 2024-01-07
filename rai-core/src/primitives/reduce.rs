use std::any::Any;

use crate::{Primitive, Shape, Tensor};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReduceSum {
    pub axes: Vec<usize>,
    pub keep_dim: bool,
}
impl ReduceSum {
    pub fn new(axes: impl Into<Vec<usize>>, keep_dim: bool) -> Self {
        Self {
            axes: axes.into(),
            keep_dim,
        }
    }
}

impl Primitive for ReduceSum {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("ReduceSum({:?}, {})", &self.axes, &self.keep_dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.reduce_sum((&self.axes, false))
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let mut shape = x.shape().to_vec();
        for axis in &self.axes {
            shape[*axis] = 1;
        }
        let cotangent_x = cotangent.reshape(shape).broadcast_to(x);
        vec![cotangent_x]
    }
}
