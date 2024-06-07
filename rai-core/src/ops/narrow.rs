use crate::{Dim, Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Narrow {
    pub dim: usize,
    pub start: usize,
    pub len: usize,
}

impl Narrow {
    pub fn new(dim: usize, start: usize, len: usize) -> Self {
        Self { dim, start, len }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn start(&self) -> usize {
        self.start
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Op for Narrow {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Narrow({}, {}, {})", self.dim, self.start, self.len)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.narrow(self.dim, self.start, self.len)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let x_dim = x.shape();
        let left_pad = if self.start == 0 {
            None
        } else {
            let mut dims = x_dim.to_vec();
            dims[self.dim] = self.start;

            Some(Tensor::zeros(dims, cotangent.dtype(), cotangent.device()))
        };
        let right_pad = x_dim[self.dim] - self.start - self.len;
        let right_pad = if right_pad == 0 {
            None
        } else {
            let mut dims = x_dim.to_vec();
            dims[self.dim] = right_pad;
            Some(Tensor::zeros(dims, cotangent.dtype(), cotangent.device()))
        };
        let cotangent_x = match (left_pad, right_pad) {
            (None, None) => cotangent.clone(),
            (Some(l), None) => Tensor::cat(&[&l, cotangent], self.dim),
            (None, Some(r)) => Tensor::cat(&[cotangent, &r], self.dim),
            (Some(l), Some(r)) => Tensor::cat(&[&l, cotangent, &r], self.dim),
        };
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn narrow(x: impl TryAsTensor, dim: impl Dim, start: usize, len: usize) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let dim = x.dim(dim);
    let device = x.device();
    let dtype = x.dtype();
    let mut shape = x.shape().to_vec();
    shape[dim] = len;
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Narrow::new(dim, start, len), inputs).into()
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn narrow<D: Dim>(&self, d: D, start: usize, len: usize) -> RaiResult<Tensor> {
        narrow(self, d, start, len)
    }
}
