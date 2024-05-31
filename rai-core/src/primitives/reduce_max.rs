use crate::{primitives::reduce_chooser_jvp_rule, Primitive, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReduceMax {
    pub dims: Vec<usize>,
    pub keep_dim: bool,
}

impl ReduceMax {
    pub fn new(dims: impl Into<Vec<usize>>, keep_dim: bool) -> Self {
        Self {
            dims: dims.into(),
            keep_dim,
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}

impl Primitive for ReduceMax {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("ReduceMax({:?}, {})", &self.dims, &self.keep_dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        reduce_chooser_jvp_rule(tangent_x, output, x, self.dims())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = if self.keep_dim {
            let mask = x.eq(output).to_dtype(x);
            let normalizer = mask.sum((self.dims(), true));
            (cotangent * mask) / normalizer
        } else {
            let mut shape = x.shape().to_vec();
            for dim in self.dims() {
                shape[*dim] = 1;
            }
            let mask = x.eq(output.reshape(&shape)).to_dtype(x);
            let normalizer = mask.sum((self.dims(), true));
            (cotangent.reshape(shape) * mask) / normalizer
        };
        vec![cotangent_x]
    }
}
