use super::ArgReduceArgs;
use crate::{Op, RaiResult, Shape, Tensor, TryAsTensor, U32};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgMax {
    pub dim: usize,
    pub keep_dim: bool,
}

impl ArgMax {
    pub fn new(dim: usize, keep_dim: bool) -> Self {
        Self { dim, keep_dim }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Op for ArgMax {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("ArgMax({}, {})", &self.dim, &self.keep_dim)
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

#[track_caller]
pub fn argmax<T: ArgReduceArgs>(x: impl TryAsTensor, args: T) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let device = x.device();
    let dtype = U32;
    let dim = x.dim(args.dim());
    let shape = x.shape_reduce([dim], args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ArgMax::new(dim, args.keep_dim()),
        inputs,
    )
    .into()
}

pub trait ArgMaxOp {
    fn argmax<T: ArgReduceArgs>(self, args: T) -> RaiResult<Tensor>;
}

impl<T> ArgMaxOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn argmax<U: ArgReduceArgs>(self, args: U) -> RaiResult<Tensor> {
        argmax(self, args)
    }
}
