use crate::{Dim, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogSoftmax {
    pub dim: usize,
}

impl LogSoftmax {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Op for LogSoftmax {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("LogSoftmax")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("LogSoftmax({:?})", &self.dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x - tangent_x.sum((self.dim, true)) * output.exp()
    }

    fn vjp(&self, output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_x = cotangent - cotangent.sum((self.dim, true)) * output.exp();
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn log_softmax<D: Dim>(x: &Tensor, d: D) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let dim = shape.dim(d);
    Tensor::new(device, dtype, shape, LogSoftmax::new(dim), inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn log_softmax<D: Dim>(&self, d: D) -> Tensor {
        log_softmax(self, d)
    }
}
