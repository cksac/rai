use std::any::Any;

use tracing::Level;

use crate::{Dims, Primitive, Shape, Tensor};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReduceSum {
    pub dims: Vec<usize>,
    pub keep_dim: bool,
}

impl ReduceSum {
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

impl Primitive for ReduceSum {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("ReduceSum({:?}, {})", &self.dims, &self.keep_dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.sum((self.dims(), false))
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let mut shape = x.shape().to_vec();
        for dim in self.dims() {
            shape[*dim] = 1;
        }
        let cotangent_x = cotangent.reshape(&shape).broadcast_to(x);
        vec![cotangent_x]
    }
}

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
        let mut shape = x.shape().to_vec();
        for dim in self.dims() {
            shape[*dim] = 1;
        }
        let mask = x.eq(output).as_type_of(x);
        let normalizer = mask.sum((self.dims(), true));
        let cotangent_x = (cotangent.reshape(shape) / normalizer) * mask;
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReduceMin {
    pub dims: Vec<usize>,
    pub keep_dim: bool,
}

impl ReduceMin {
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

impl Primitive for ReduceMin {
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
        let mut shape = x.shape().to_vec();
        for dim in self.dims() {
            shape[*dim] = 1;
        }
        let mask = x.eq(output).as_type_of(x);
        let normalizer = mask.sum((self.dims(), true));
        let cotangent_x = (cotangent.reshape(shape) / normalizer) * mask;
        vec![cotangent_x]
    }
}

fn reduce_chooser_jvp_rule(g: &Tensor, ans: &Tensor, operand: &Tensor, dims: &[usize]) -> Tensor {
    let mut shape = operand.shape().to_vec();
    for dim in dims {
        shape[*dim] = 1;
    }
    let location_indicators = operand.eq(ans.reshape(shape)).as_type_of(g);
    let counts = location_indicators.sum(dims);
    (g * location_indicators).sum(dims) / counts
}
