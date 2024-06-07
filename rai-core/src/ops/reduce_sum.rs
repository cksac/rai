use crate::{Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

use super::ReduceArgs;

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

impl Op for ReduceSum {
    fn clone_boxed(&self) -> Box<dyn Op> {
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
        let cotangent_x = if self.keep_dim {
            cotangent.broadcast_to(x)
        } else {
            let mut shape = x.shape().to_vec();
            for dim in self.dims() {
                shape[*dim] = 1;
            }
            cotangent.reshape(&shape).broadcast_to(x)
        };
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn sum<T: ReduceArgs>(x: impl TryAsTensor, args: T) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let device = x.device();
    let dtype = x.dtype();
    let dims = x.dims(args.dims());
    let shape = x.shape_reduce(&dims, args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ReduceSum::new(dims, args.keep_dim()),
        inputs,
    )
    .into()
}

pub trait ReduceSumOp {
    fn sum<T: ReduceArgs>(self, args: T) -> RaiResult<Tensor>;
}

impl<T> ReduceSumOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn sum<U: ReduceArgs>(self, args: U) -> RaiResult<Tensor> {
        sum(self, args)
    }
}
