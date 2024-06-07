use crate::{ops::reduce_chooser_jvp_rule, Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

use super::ReduceArgs;

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

impl Op for ReduceMax {
    fn clone_boxed(&self) -> Box<dyn Op> {
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

#[track_caller]
pub fn max<T: ReduceArgs>(x: impl TryAsTensor, args: T) -> RaiResult<Tensor> {
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
        ReduceMax::new(dims, args.keep_dim()),
        inputs,
    )
    .into()
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn max<A: ReduceArgs>(&self, args: A) -> RaiResult<Tensor> {
        max(self, args)
    }
}
