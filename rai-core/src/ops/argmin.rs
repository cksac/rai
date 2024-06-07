use crate::{Op, RaiResult, Shape, Tensor, TryAsTensor, U32};
use std::any::Any;
use tracing::Level;

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
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("ArgMin({}, {})", &self.dim, &self.keep_dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> RaiResult<Tensor> {
        output.zeros_like()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(
        &self,
        output: &Tensor,
        primals: &[Tensor],
        cotangent: &Tensor,
    ) -> RaiResult<Vec<Tensor>> {
        Ok(vec![]).into()
    }
}

#[track_caller]
pub fn argmin<T: ArgReduceArgs>(x: impl TryAsTensor, args: T) -> RaiResult<Tensor> {
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
        ArgMin::new(dim, args.keep_dim()),
        inputs,
    )
    .into()
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn argmin<U: ArgReduceArgs>(&self, args: U) -> RaiResult<Tensor> {
        argmin(self, args)
    }
}
