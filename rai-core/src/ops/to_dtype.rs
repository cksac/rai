use crate::{AsDType, Op, RaiResult, Shape, Tensor, TryAsTensor, Type};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ToDType<D: Type> {
    pub dtype: D,
}

impl<D: Type> ToDType<D> {
    pub fn new(dtype: D) -> Self {
        Self { dtype }
    }
}

impl<D: Type> Op for ToDType<D> {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("ToDType({:?})", &self.dtype)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.to_dtype(self.dtype)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent.to_dtype(x);
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn to_dtype(x: impl TryAsTensor, dtype: impl AsDType) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let dtype = dtype.dtype();
    if x.dtype() == dtype {
        return x.clone();
    }
    let device = x.device();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let op = dtype.to_dtype_op();
    Tensor::new(device, dtype, shape, op, inputs).into()
}

pub trait ToDTypeOp {
    fn to_dtype<D: Type>(self, dtype: D) -> RaiResult<Tensor>;
}

impl<T> ToDTypeOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn to_dtype<D: Type>(self, dtype: D) -> RaiResult<Tensor> {
        to_dtype(self, dtype)
    }
}
