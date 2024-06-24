use crate::{Op, Shape, Tensor, U32};
use std::{any::Any, borrow::Cow};

use super::ArgReduceArgs;

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

impl Op for ArgMin {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("ArgMin")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("ArgMin({}, {})", &self.dim, &self.keep_dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        output.zeros_like()
    }

    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        vec![]
    }
}

#[track_caller]
pub fn argmin<T: ArgReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let device = x.device();
    let dtype = U32;
    let dim = x.dim(args.dim());
    let shape = x.shape_reduce([dim], args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ArgMin::new(dim, args.keep_dim()),
        inputs,
    )
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn argmin<T: ArgReduceArgs>(&self, args: T) -> Tensor {
        argmin(self, args)
    }
}
