use crate::{Dims, Op, RaiResult, Shape, Tensor, TryAsTensor};
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
pub fn permute(x: impl TryAsTensor, d: impl Dims<Vec<usize>>) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let dims = x.dims(d);
    assert_eq!(dims.len(), x.rank());
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.sizes(&dims);
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Permute::new(dims), inputs).into()
}

pub trait PermuteOp {
    fn permute<D: Dims<Vec<usize>>>(self, d: D) -> RaiResult<Tensor>;
}

impl<T> PermuteOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn permute<D: Dims<Vec<usize>>>(self, d: D) -> RaiResult<Tensor> {
        permute(self, d)
    }
}
