use crate::{Dims, Op, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permute {
    pub dims: Vec<usize>,
}
impl Permute {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}

impl Op for Permute {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Permute({:?})", self.dims)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.permute(self.dims())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let dims = self.dims();
        let mut inv_dims = vec![0; dims.len()];
        for (i, &dim_idx) in dims.iter().enumerate() {
            inv_dims[dim_idx] = i
        }
        let cotangent_x = cotangent.permute(inv_dims);
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn permute(x: &Tensor, d: impl Dims<Vec<usize>>) -> Tensor {
    let dims = x.dims(d);
    assert_eq!(dims.len(), x.rank());
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.sizes(&dims);
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Permute::new(dims), inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn permute(&self, d: impl Dims<Vec<usize>>) -> Tensor {
        permute(self, d)
    }
}
