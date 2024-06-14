use crate::{Error, Op, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reshape {
    pub shape: Vec<usize>,
}
impl Reshape {
    pub fn new(shape: impl Shape) -> Self {
        Self {
            shape: shape.shape().to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl Op for Reshape {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Reshape({:?})", self.shape)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.reshape(self.shape())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent.reshape(x);
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn reshape(x: &Tensor, shape: impl Shape) -> Tensor {
    if x.shape() == shape.shape() {
        return x.clone();
    }
    let device = x.device();
    let dtype = x.dtype();
    let inputs = vec![x.clone()];
    if x.elem_count() == shape.elem_count() {
        Tensor::new(
            device,
            dtype,
            shape.shape().to_owned(),
            Reshape::new(&shape),
            inputs,
        )
    } else {
        Tensor::err(
            device,
            dtype,
            shape.shape().to_owned(),
            Reshape::new(&shape),
            inputs,
            Error::IncompatibleShape {
                lhs: x.shape().to_owned(),
                rhs: shape.shape().to_owned(),
            },
        )
    }
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn reshape(&self, shape: impl Shape) -> Tensor {
        reshape(self, shape)
    }
}
